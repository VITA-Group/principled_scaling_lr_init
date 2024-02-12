'''
    main process for training a network
'''
import os
import random
import time
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data

import models
from dag_utils import build_model, dag2affinity, effective_depth_width, dag_depths
from dataset import cifar10_dataloaders, cifar100_dataloaders, imagenet_dataloaders, imagenet_16_120_dataloaders
from logger import prepare_seed, prepare_logger
from utils import save_checkpoint, warmup_lr, AverageMeter, accuracy

from pdb import set_trace as bp

__all__ = ["auto_scale_lr_depth_width"]

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='mlp', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: mlp)')
parser.add_argument('--dag', type=str, default=None, help='from-to edges separated by underscore. 0: broken edge; 1: skip-connect; 2: linear or conv')
parser.add_argument('--dags', type=str, nargs='+', default=None, help='from-to edges separated by underscore. 0: broken edge; 1: skip-connect; 2: linear or conv')
parser.add_argument('--width', type=int, default=None, help='hidden width (for mlp, cnn)')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel sizel')
parser.add_argument('--act', type=str, default='relu', choices=['relu', 'gelu'])

##################################### General setting ############################################
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--inference', action="store_true", help="testing")
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='./experiment', type=str)
parser.add_argument('--exp_name', help='additional names for experiment', default='', type=str)
parser.add_argument('--repeat', default=1, type=int, help='repeat training of DAG w. different random seed')

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.46, type=float, help='initial learning rate', required=True)
parser.add_argument('--lr_autoscale', action="store_true", help="automatically re-scale LR")
parser.add_argument('--momentum', default=0., type=float, help='momentum') # 0.9
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay') # 1e-4
parser.add_argument('--epochs', default=None, type=int, help='number of total epochs to run')
parser.add_argument('--steps', default=None, type=int, help='number of total steps to run (i.e. steps within the 1st epoch)')
parser.add_argument('--nesterov', action="store_true", help="use nesterov")
parser.add_argument('--aug', action="store_true", help="use augmentation")
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--decreasing_lr', action="store_true", help='decreasing strategy')


def auto_scale_lr_depth_width(lr_base, depths_base, depths):
    return lr_base * ((np.array(depths)**3).sum() ** -0.5) / ((np.array(depths_base)**3).sum() ** -0.5)


code012_201 = {
    0: 0,
    1: 1,
    2: 2,
    3: 2,
    4: 1
}


def main():

    global args
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(0, 999)

    torch.cuda.set_device(int(args.gpu))

    if args.dataset in ['cifar10', 'cifar100']:
        from torchvision.datasets import CIFAR10, CIFAR100
        CIFAR10(args.data, train=True, download=True)
        CIFAR10(args.data, train=False, download=True)
        CIFAR100(args.data, train=True, download=True)
        CIFAR100(args.data, train=False, download=True)

    # instead of training models in a sequential order, we train over random_dag_list to simulate samplings over the model space
    if args.arch in ['mlp', 'cnn']:
        DEPTHS_BASE = [3] # stem-ReLU -> linear-ReLU -> output layer
        KERNEL_BASE = 3
        with open('all_dags_str_N5.json') as json_file:
            all_dags_str = json.load(json_file)
        if args.dags:
            random_dag_list = []
            for _dag in args.dags:
                if _dag in all_dags_str:
                    random_dag_list.append(all_dags_str.index(_dag))
                else:
                    all_dags_str.append(_dag)
                    random_dag_list.append(len(all_dags_str)-1)
        else:
            random_dag_list = np.load("random_dag_list_N5.npy")
    elif args.arch.startswith('tinynetwork'):
        DEPTHS_BASE = [1] # 1_01_001
        with open('arch_code_201.json') as json_file: # dag code
            all_dags_str = json.load(json_file)
        if args.dags:
            random_dag_list = []
            for _dag in args.dags:
                random_dag_list.append(all_dags_str.index(_dag))
        else:
            random_dag_list = np.load("arch_indice_201.npy") # random list of index
    else:
        all_dags_str = None
        random_dag_list = [args.arch]

    if args.steps:
        args.epochs = 1

    timestamp = "{:}".format(time.strftime('%h-%d-%Y-%C_%H-%M-%S', time.gmtime(time.time())))
    job_name = "BULK-{dataset}-{arch}{kernel}{width}-LR{lr}{scale}{nesterov}-BS{batch_size}-Epoch{epoch}{exp_name}-{timestamp}".format(
        dataset=args.dataset + (".Aug" if args.aug else ""),
        arch=args.arch + (".GeLU" if args.act == 'gelu' else ""),
        kernel=".K%d"%args.kernel_size if args.arch == "cnn" else "",
        width=".W%d"%args.width if args.width else "",
        lr="%f"%args.lr,
        scale=".AutoScale" if args.lr_autoscale else "",
        nesterov=".Nev" if args.nesterov else "",
        batch_size=args.batch_size,
        epoch=str(args.epochs) + (".Steps%d"%args.steps if args.steps else ""),
        exp_name="" if args.exp_name == "" else "-"+args.exp_name,
        timestamp=timestamp
    )
    SAVE_DIR = os.path.join(args.save_dir, job_name)

    PID = os.getpid()
    print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, SAVE_DIR))

    pbar = tqdm(random_dag_list, position=0, leave=True)
    for dag_idx in pbar:
        if args.arch in ['mlp', 'cnn', 'tinynetwork']:
            args.dag = all_dags_str[dag_idx]
        if not os.path.exists(os.path.join(SAVE_DIR, "%d"%dag_idx)):
            os.makedirs(os.path.join(SAVE_DIR, "%d"%dag_idx))
        with open(os.path.join(SAVE_DIR, "%d"%dag_idx, "dag_str.txt"), 'w') as f:
            f.write("%s"%args.dag)
        if args.lr_autoscale:
            if args.arch.startswith("tinynetwork"):
                _aff = dag2affinity([ [code012_201[int(value)] for value in node] for node in all_dags_str[dag_idx].split("_")])
            else:
                _aff = dag2affinity([ [int(value) for value in node] for node in all_dags_str[dag_idx].split("_")])
            if args.arch == "tinynetwork":
                _paths = effective_depth_width(_aff)[2]
                _depths = [1+sum([_p > 0 for _p in _path]) for _path in _paths] # compensate for skip connection
                if len(_depths) == 0 or sum(_depths) == 0:
                    print("Bad architecture: no valid end-to-end path, depth = 0. Exit.")
                    exit(0)
                lr = auto_scale_lr_depth_width(args.lr, DEPTHS_BASE, _depths)
            elif args.arch in ["mlp", "cnn"]:
                _depths = np.array(dag_depths(_aff))
                lr = auto_scale_lr_depth_width(args.lr, DEPTHS_BASE, _depths+DEPTHS_BASE[0])
                if args.arch == "cnn":
                    lr = lr * (KERNEL_BASE / args.kernel_size)
        for r_idx in range(args.repeat):
            seed = args.seed + r_idx

            if args.arch in ['mlp', 'cnn', 'tinynetwork']:
                args.save_dir = os.path.join(SAVE_DIR, "%d"%dag_idx, "%f"%lr, "%d"%seed)
            else:
                args.save_dir = os.path.join(SAVE_DIR, "%f"%lr, "%d"%seed)

            prefix = "Train"

            prepare_seed(seed)
            pbar.set_description("%s %s depths=%s LR=%f seed %d"%(prefix, dag_idx, str(_depths), lr, seed))
            train_model(dag_idx, lr)


def train_model(dag_idx, lr):
    global args
    logger = prepare_logger(args, verbose=False)

    if not args.inference:
        os.makedirs(args.save_dir, exist_ok=True)

    # prepare dataset
    NUM_VAL_IMAGE = 50
    c_in = 3
    if args.dataset == 'cifar10':
        classes = 10
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = cifar10_dataloaders(
            batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,
            aug=args.aug,
            flatten=args.arch == "mlp",
        )
    elif args.dataset == 'cifar100':
        classes = 100
        dummy_shape = (3, 32, 32)
        train_loader, val_loader, test_loader = cifar100_dataloaders(
            batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,
            aug=args.aug,
            flatten=args.arch == "mlp",
        )
    elif args.dataset == 'imagenet16_120':
        classes = 120
        dummy_shape = (3, 16, 16)
        train_loader, val_loader, test_loader = imagenet_16_120_dataloaders(
            batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,
            aug=args.aug,
            flatten=args.arch == "mlp",
        )
    elif args.dataset == 'tinyimagenet':
        classes = 200
        dummy_shape = (3, 64, 64)
        train_loader, val_loader, test_loader = imagenet_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset == 'imagenet':
        classes = 1000
        dummy_shape = (3, 224, 224)
        train_loader, val_loader, test_loader = imagenet_dataloaders(batch_size = args.batch_size, img_shape=dummy_shape[1], data_dir = args.data, num_workers = args.workers, flatten=args.arch == "mlp")
    elif args.dataset is None:
        pass
    else:
        raise ValueError('Dataset not supprot yet!')

    model = build_model(args, classes, dummy_shape)
    logger.log(str(model))

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = None
    if args.decreasing_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    all_result = {}
    all_result['train_acc'] = []
    all_result['test_acc'] = []
    all_result['val_acc'] = []
    all_result['train_loss'] = []
    all_result['test_loss'] = []
    all_result['val_loss'] = []
    start_epoch = 0

    logger.log("Path {}".format(args.save_dir))
    loss_steps_epochs = []
    for epoch in range(start_epoch, args.epochs):

        train_loss, train_acc, loss_steps = train(train_loader, model, criterion, optimizer, epoch, args.steps)
        loss_steps_epochs += list(loss_steps)
        logger.writer.add_scalar("train/loss", train_loss, epoch)
        logger.writer.add_scalar("train/accuracy", train_acc, epoch)
        pbar_str = "Epoch:{} Train:{:.2f} (Loss:{:.4f}) ".format(epoch, train_acc, train_loss)
        all_result['train_acc'].append(train_acc)
        all_result['train_loss'].append(train_loss)
        if val_loader:
            val_loss, val_acc = validate(val_loader, model, criterion)
            logger.writer.add_scalar("validation/loss", val_loss, epoch)
            logger.writer.add_scalar("validation/accuracy", val_acc, epoch)
            pbar_str += "Validation:{:.2f} (Loss:{:.4f})".format(val_acc, val_loss)
            all_result['val_acc'].append(val_acc)
            all_result['val_loss'].append(val_loss)
            plt.plot(all_result['val_acc'], label='val_acc')
        if test_loader:
            test_loss, test_acc = validate(test_loader, model, criterion)
            logger.writer.add_scalar("test/loss", test_loss, epoch)
            logger.writer.add_scalar("test/accuracy", test_acc, epoch)
            pbar_str += "Test:{:.2f} (Loss:{:.4f})".format(test_acc, test_loss)
            all_result['test_acc'].append(test_acc)
            all_result['test_loss'].append(test_loss)
            plt.plot(all_result['test_acc'], label='test_acc')
        logger.log("Path {}".format(args.save_dir))
        logger.log(pbar_str)
        if scheduler:
            logger.log("LR:{}".format(scheduler.get_last_lr()[0]))

        if scheduler: scheduler.step()

        checkpoint = {
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'train_losses': loss_steps_epochs
        }
        save_checkpoint(checkpoint, is_best=False, save_path=args.save_dir)

        # plot training curve
        plt.plot(all_result['train_acc'], label='train_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

        if np.isnan(train_loss):
            all_result['train_acc'] += [train_acc] * (args.epochs - epoch - 1)
            all_result['train_loss'] += [train_loss] * (args.epochs - epoch - 1)
            break

    if val_loader:
        val_pick_best_epoch = np.argmax(np.array(all_result['val_acc']))
    else:
        val_pick_best_epoch = len(all_result['train_acc']) - 1
    if test_loader:
        best_acc = all_result['test_acc'][val_pick_best_epoch]
        best_loss = all_result['test_loss'][val_pick_best_epoch]
    else:
        best_acc = all_result['val_acc'][val_pick_best_epoch]
        best_loss = all_result['val_loss'][val_pick_best_epoch]
    logger.log('* best accuracy = {}, best loss = {}, Epoch = {}'.format(best_acc, best_loss, val_pick_best_epoch+1))
    checkpoint = {
        'result': all_result,
        'epoch': epoch + 1,
        # 'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'best_loss': best_loss,
        'best_epoch': val_pick_best_epoch,
        # 'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict() if scheduler else None,
        'train_losses': loss_steps_epochs
    }
    save_checkpoint(checkpoint, is_best=False, save_path=args.save_dir)


def train(train_loader, model, criterion, optimizer, epoch, steps):
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_steps = []

    # switch to train mode
    model.train()

    for i, (image, target) in enumerate(train_loader):
        if isinstance(steps, int) and epoch == 0 and i >= steps: break
        if epoch < args.warmup:
            warmup_lr(args.warmup, args.lr, epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        prec1, gt_num = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], gt_num[0])
        loss_steps.append(float(loss.item()))

        losses.update(loss.item(), image.size(0))

    return float(losses.avg), float(top1.vec2sca_avg), loss_steps


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        prec1, gt_num = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], gt_num[0])
        losses.update(loss.item(), image.size(0))

    return float(losses.avg), float(top1.vec2sca_avg)


if __name__ == '__main__':
    main()
