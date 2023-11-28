# Mixed Barlow Twins

> Official Pytorch implementation for Mixed Barlow Twins on small datasets (CIFAR10, STL10, and Tiny ImageNet) and ImageNet.

Correspondence to: 
  - Wele Gedara Chaminda Bandara [wgcban](https://www.wgcban.com)

##
[**Guarding Barlow Twins Against Overfitting with Mixed Samples**](https://arxiv.org/pdf/)<br>
[Wele Gedara Chaminda Bandara](https://www.wgcban.com) (Johns Hopkins University), [Celso M. De Melo](https://celsodemelo.net) (U.S. Army Research Laboratory), and [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/) (Johns Hopkins University) <br>

I hope this work will be useful for your research :smiling_face_with_three_hearts: 

## Usage

### Disclaimer
A large portion of the code is from [Barlow Twins HSIC](https://github.com/yaohungt/Barlow-Twins-HSIC) (for experiments on small datasets: Cifar10, Cifar100, TinyImageNet, and STL-10) and official implementation of Barlow Twins [here](https://github.com/facebookresearch/barlowtwins) (for experiments on ImageNet), which is a great resource for academic development.

Also, note that the implementation of SOTA methods (SimCLR, BYOL, and Witening-MSE) in `ssl-sota` are copied from [Witening-MSE](https://github.com/htdt/self-supervised).

### Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
- thop
```
pip install thop
```

- clone this repo (recursively). 
With version 2.13 of Git and later,
```
git clone https://github.com/wgcban/ssl-aug.git --recurse-submodules
```
or
```
git clone https://github.com/wgcban/ssl-aug.git --recursive
```


### Supported Datasets
`Cifar10`, `Cifar100`, `STL10`, [`Tiny_ImageNet`](https://github.com/rmccorm4/Tiny-Imagenet-200), STL-10, and ImageNet. We use `Cifar10`, `Cifar100`, and `STL10` datasets directly available in PyTorch. We pre-process the TinyImageNet accroding to the script given [here](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4). For ImageNet experiments, we follow the exact steps used in original Barlow Twins implementation.

### Supported SOTA methods
- SimCLR: contrastive learning for SSL 
- BYOL: distilation for SSL
- Witening MSE: infomax for SSL
- Barlow Twins: infomax for SSL
- Mixed Barlow Twins (ours): infomax + mixup samples for SSL

### Pre-training and k-NN evaluation results with Mixed Barlow Twins / Barlow Twins (is_mixup = False)
Train and k-NN Evaluation using Mixed Barlow Twins on `Cifar10`, `Cifar100`, `TinyImageNet`, and `STL-10`:
On `Cifar10`:
```
sh scripts-pretrain/cifar10.sh
```
On `Cifar100`:
```
sh scripts-pretrain/cifar100.sh
```
On `tinyimagenet`:
```
sh scripts-pretrain/tinyimagenet.sh
```
On `stl-10`:
```
sh scripts-pretrain/stl10.sh
```
On `ImageNet`:
```
sh scripts-pretrain/imagenet.sh
```

### Linear Evaluation with Mixed Barlow Twins
**Make sure to specify `model_path` argument correctly.**
On `Cifar10`:
```
sh scripts-linear/cifar10.sh
```
On `Cifar100`:
```
sh scripts-linear/cifar100.sh
```
On `tinyimagenet`:
```
sh scripts-linear/tinyimagenet.sh
```
On `stl-10`:
```
sh scripts-linear/stl10.sh
```
On `ImageNet`:
```
sh scripts-linear/imagenet_sup.sh
```

### Reference
Please consider citing our work, if you feel our work is useful in your work. Thanks.
```
```