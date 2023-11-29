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

### Pre-training and k-NN evaluation results with Mixed Barlow Twins (is_mixup = True) / Barlow Twins (is_mixup = False)
Train and k-NN Evaluation using Mixed Barlow Twins on `Cifar10`, `Cifar100`, `TinyImageNet`, and `STL-10`:
On `Cifar10`, `Cifar100`, `tinyimagenet`, `stl-10`, `ImageNet`:
```
sh scripts-pretrain/[cifar10].sh
```
```
sh scripts-pretrain/cifar100.sh
```
```
sh scripts-pretrain/tinyimagenet.sh
```
```
sh scripts-pretrain/stl10.sh
```
```
sh scripts-pretrain/imagenet.sh
```

### Linear Evaluation with Mixed Barlow Twins
**Make sure to specify `model_path` argument correctly.**
On `Cifar10`, `Cifar100`, `tinyimagenet`, `stl-10`, and `ImageNet`:
```
sh scripts-linear/cifar10.sh
```
```
sh scripts-linear/cifar100.sh
```
```
sh scripts-linear/tinyimagenet.sh
```
```
sh scripts-linear/stl10.sh
```
```
sh scripts-linear/imagenet_sup.sh
```

# Pre-trained model checkpoints
The following links provide pre-trained models:
## ResNet-18
| Dataset        |  d   | Lambda_BT | Lambda_Reg | Path to Pretrained Model | KNN Acc. | Linear Acc. |
| ----------     | ---  | ---------- | ---------- | ------------------------ | -------- | ----------- |
| CIFAR-10       | 1024 | 0.0078125  | 4.0        | 4wdhbpcf_0.0078125_1024_256_cifar10_model.pth     | 90.52    | 92.58        |
| CIFAR-100      | 1024 | 0.0078125  | 4.0        | 76kk7scz_0.0078125_1024_256_cifar100_model.pth     | 61.25     | 69.31        |
| TinyImageNet   | 1024 | 0.0009765  | 4.0        | 02azq6fs_0.0009765_1024_256_tiny_imagenet_model.pth     | 38.11    | 51.67        |
| STL-10         | 1024 | 0.0078125  | 2.0        | i7det4xq_0.0078125_1024_256_stl10_model.pth     | 88.94     | 91.02        |



## ResNet-50
| Dataset        |  d   | Lambda_BT | Lambda_Reg | Path to Pretrained Model | KNN Acc. | Linear Acc. |
| ----------     | ---  | ---------- | ---------- | ------------------------ | -------- | ----------- |
| CIFAR-10       | 1024 | 0.0078125  | 4.0        | v3gwgusq_0.0078125_1024_256_cifar10_model.pth     | 91.39     | 93.89        |
| CIFAR-100      | 1024 | 0.0078125  | 4.0        | z6ngefw7_0.0078125_1024_256_cifar100_model_2000.pth     | 64.32     | 72.51        |
| TinyImageNet   | 1024 | 0.0009765  | 4.0        | kxlkigsv_0.0009765_1024_256_tiny_imagenet_model_2000.pth     | 42.21     | 51.84        |
| STL-10         | 1024 | 0.0078125  | 2.0        | pbknx38b_0.0078125_1024_256_stl10_model.pth     | 87.79     | 91.70        |
| ImageNet       | 1024 | 0.0051  | 0.1        | 13awtq23_0.0051_8192_1024_imagenet_0.1_resnet50.pth     | -     | 72.1        |


### Reference
Please consider citing our work, if you feel our work is useful in your work. Thanks.
```
```