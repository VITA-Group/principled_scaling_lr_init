# Principled Architecture-aware Scaling of Hyperparameters [[PDF]()]

[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Wuyang Chen, Junru Wu, Zhangyang Wang, Boris Hanin

In ICLR 2024.


## Overview

In this work, we precisely characterize the dependence of initializations and maximal learning rates on the network architecture, which includes the network depth, width, convolutional kernel size, and connectivity patterns.
We can achieve zero-shot scaling of initialization and learning rates across MLPs and CNNs with sophisticated graph topologies.

Highlights:
* A simple architecture-aware fan-in initialization scheme that can provably preserve the flow of information through any architecture’s graph topology.
* For fully-connected and convolutional layers, we analytically compute the dependence on the architecture for how to scale learning rates in order to achieve the maximal update (μP) heuristic [((Yang et al., 2022)](https://arxiv.org/abs/2203.03466)
* In experiments, we not only verify the superior performance of our prescriptions, but also re-evaluate the quality of standard architecture benchmarks (by unleashing the potential of architectures with architecture-aware initializations and learning rates).


## Prerequisites
- Ubuntu 18.04
- Python 3.6.9
- CUDA 11.1 (lower versions may work but were not tested)
- NVIDIA GPU + CuDNN v8.0

This repository has been tested on RTX A6000. Configurations may need to be changed on different platforms.


## Installation
* Clone this repo:
```bash
git clone https://github.com/chenwydj/principled_hp_scaling.git
cd principled_hp_scaling
```
* Install dependencies:
```bash
pip install -r requirements.txt
```

If you need to train on ImageNet-16-120, please refer to the GitHub repo of [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)

## Usage
1. Find the base maximal learning rate of the shallow version of your target model (by reducing the number of hidden layers to 1: "input_layer->hidden_layer->output_layer")
2. Initialize your target network (of deeper layers or different topologies) according to our equation 4 (see our example for [MLPs](), [CNNs](), and [architectures with complicated graph topologies]()).
3. Train your target network by scaling the learning rate based on equation 8 for MLPs or equation 9 for CNNs (see our [scaling rule]()).


```bash
python main.py \
--data /path/to/dataset \
--gpu 0 \
--batch_size 256 \
--aug \ # use augmentations
--momentum 0.9 \
--weight_decay 0.0005 \
--nesterov \
--decreasing_lr \ # classic learning rate scheduler (e.g. cosine)
--width 16 \
--arch tinynetwork \ # architectures on NAS-Bench-201; or "mlp", "cnn"
--dataset cifar10 \ # or cifar100, imagenet16_120
--epochs 200 \
--lr 0.46 \ # base maximal learning rate found for the shallow version of your target model
--lr_autoscale # principled scaling of the base lr to your target network
```
