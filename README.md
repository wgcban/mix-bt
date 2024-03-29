# Mixed Barlow Twins for Self-Supervised Representation Learning
>[**Guarding Barlow Twins Against Overfitting with Mixed Samples**](https://arxiv.org/abs/2312.02151)<br>

>[![arXiv](https://img.shields.io/badge/arXiv-2312.02151-b31b1b)](https://arxiv.org/abs/2312.02151)
>[![Hugging Face Model Card](https://img.shields.io/badge/Model%20Card-Hugging%20Face-%2334D058)](https://huggingface.co/wgcban/mix-bt)


[Wele Gedara Chaminda Bandara](https://www.wgcban.com) (Johns Hopkins University), [Celso M. De Melo](https://celsodemelo.net) (U.S. Army Research Laboratory), and [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/) (Johns Hopkins University) <br>

## 1 Overview of Mixed Barlow Twins

TL;DR
- Mixed Barlow Twins aims to improve sample interaction during Barlow Twins training via linearly interpolated samples. 
- We introduce an additional regularization term to the original Barlow Twins objective, assuming linear interpolation in the input space translates to linearly interpolated features in the feature space.
- Pre-training with this regularization effectively mitigates feature overfitting and further enhances the downstream performance on `CIFAR-10`, `CIFAR-100`, `TinyImageNet`, `STL-10`, and `ImageNet` datasets.

<img src="figs/mix-bt.svg" width="1024">

$C^{MA} = (Z^M)^TZ^A$

$C^{MB} = (Z^M)^TZ^B$

$C^{MA}_{gt} = \lambda (Z^A)^TZ^A + (1-\lambda)\mathtt{Shuffle}^*(Z^B)^TZ^A$

$C^{MB}_{gt} = \lambda (Z^A)^TZ^B + (1-\lambda)\mathtt{Shuffle}^*(Z^B)^TZ^B$

## 2 Usage
### 2.1 Requirements

Before using this repository, make sure you have the following prerequisites installed:

- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)

You can install PyTorch with the following [command](https://pytorch.org/get-started/locally/) (in Linux OS):
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2.2 Installation

To get started, clone this repository:
```bash
git clone https://github.com/wgcban/mix-bt.git
```

Next, create the [conda](https://docs.conda.io/projects/conda/en/stable/) environment named `ssl-aug` by executing the following command:
```bash
conda env create -f environment.yml
```

All the train-val-test statistics will be automatically upload to [`wandb`](https://wandb.ai/home), and please refer [`wandb-quick-start`](https://wandb.ai/quickstart?utm_source=app-resource-center&utm_medium=app&utm_term=quickstart) documentation if you are not familiar with using `wandb`. 

### 2.3 Supported Pre-training Datasets

This repository supports the following pre-training datasets:
- `CIFAR-10`: https://www.cs.toronto.edu/~kriz/cifar.html
- `CIFAR-100`: https://www.cs.toronto.edu/~kriz/cifar.html
- `Tiny-ImageNet`: https://github.com/rmccorm4/Tiny-Imagenet-200
- `STL-10`: https://cs.stanford.edu/~acoates/stl10/
- `ImageNet`: https://www.image-net.org

`CIFAR-10`, `CIFAR-100`, and `STL-10` datasets are directly available in PyTorch. 

To use `TinyImageNet`, please follow the preprocessing instructions provided in the [TinyImageNet-Script](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4). Download these datasets and place them in the `data` directory.

### 2.4 Supported Transfer Learning Datasets
You can download and place transfer learning datasets under their respective paths, such as 'data/DTD'. The supported transfer learning datasets include:
- `DTD`: https://www.robots.ox.ac.uk/~vgg/data/dtd/ 
- `MNIST`: http://yann.lecun.com/exdb/mnist/
- `FashionMNIST`: https://github.com/zalandoresearch/fashion-mnist
- `CUBirds`: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- `VGGFlower`: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- `Traffic Signs`: https://benchmark.ini.rub.de/gtsdb_dataset.html
- `Aircraft`: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

### 2.5 Supported SSL Methods

This repository supports the following Self-Supervised Learning (SSL) methods:

- [`SimCLR`](https://arxiv.org/abs/2002.05709): contrastive learning for SSL 
- [`BYOL`](https://arxiv.org/abs/2006.07733): distilation for SSL
- [`Witening MSE`](http://proceedings.mlr.press/v139/ermolov21a/ermolov21a.pdf): infomax for SSL
- [`Barlow Twins`](https://arxiv.org/abs/2103.03230): infomax for SSL
- **`Mixed Barlow Twins (ours)`**: infomax + mixed samples for SSL

### 2.6 Pre-Training with Mixed Barlow Twins
To start pre-training and obtain k-NN evaluation results for Mixed Barlow Twins on `CIFAR-10`, `CIFAR-100`, `TinyImageNet`, and `STL-10` with `ResNet-18/50` backbones, please run:
```bash
sh scripts-pretrain-resnet18/[dataset].sh
```
```bash
sh scripts-pretrain-resnet50/[dataset].sh
```

To start the pre-training on `ImageNet` with `ResNet-50` backbone, please run:
```bash
sh scripts-pretrain-resnet18/imagenet.sh
```

### 2.7 Linear Evaluation of Pre-trained Models
Before running linear evaluation, *ensure that you specify the `model_path` argument correctly in the corresponding .sh file*. 

To obtain linear evaluation results on `CIFAR-10`, `CIFAR-100`, `TinyImageNet`, `STL-10` with `ResNet-18/50` backbones, please run:
```bash
sh scripts-linear-resnet18/[dataset].sh
```
```bash
sh scripts-linear-resnet50/[dataset].sh
```

To obtain linear evaluation results on `ImageNet` with `ResNet-50` backbone, please run:
```bash
sh scripts-linear-resnet50/imagenet_sup.sh
```


### 2.8 Transfer Learning of Pre-trained Models
To perform transfer learning from pre-trained models on `CIFAR-10`, `CIFAR-100`, and `STL-10` to fine-grained classification datasets, execute the following command, making sure to specify the `model_path` argument correctly:
```bash
sh scripts-transfer-resnet18/[dataset]-to-x.sh
```

## 3 Pre-Trained Checkpoints
Download the pre-trained models from [GitHub (Releases v1.0.0)](https://github.com/wgcban/mix-bt/releases/tag/v1.0.0) and store them in `checkpoints/`. This repository provides pre-trained checkpoints for both [`ResNet-18`](https://arxiv.org/abs/1512.03385) and [`ResNet-50`](https://arxiv.org/abs/1512.03385) architectures.

#### 3.1 ResNet-18 \[`CIFAR-10`, `CIFAR-100`, `TinyImageNet`, and `STL-10`\]
| Dataset        |  $d$   | $\lambda_{BT}$ | $\lambda_{reg}$ | Download Link to Pretrained Model | KNN Acc. | Linear Acc. |
| ----------     | ---  | ---------- | ---------- | ------------------------ | -------- | ----------- |
| `CIFAR-10`       | 1024 | 0.0078125  | 4.0        | [4wdhbpcf_cifar10.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/4wdhbpcf_0.0078125_1024_256_cifar10_model.pth)     | 90.52    | 92.58        |
| `CIFAR-100`     | 1024 | 0.0078125  | 4.0        | [76kk7scz_cifar100.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/76kk7scz_0.0078125_1024_256_cifar100_model.pth)     | 61.25     | 69.31        |
| `TinyImageNet`   | 1024 | 0.0009765  | 4.0        | [02azq6fs_tiny_imagenet.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/02azq6fs_0.0009765_1024_256_tiny_imagenet_model.pth)     | 38.11    | 51.67        |
| `STL-10`        | 1024 | 0.0078125  | 2.0        | [i7det4xq_stl10.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/i7det4xq_0.0078125_1024_256_stl10_model.pth)     | 88.94     | 91.02        |

#### 3.2 ResNet-50 \[`CIFAR-10`, `CIFAR-100`, `TinyImageNet`, and `STL-10`\]
| Dataset        |  $d$   | $\lambda_{BT}$ | $\lambda_{reg}$ | Download Link to Pretrained Model | KNN Acc. | Linear Acc. |
| ----------     | ---  | ---------- | ---------- | ------------------------ | -------- | ----------- |
| `CIFAR-10`       | 1024 | 0.0078125  | 4.0        | [v3gwgusq_cifar10.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/v3gwgusq_0.0078125_1024_256_cifar10_model.pth)     | 91.39     | 93.89        |
| `CIFAR-100`      | 1024 | 0.0078125  | 4.0        | [z6ngefw7_cifar100.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/z6ngefw7_0.0078125_1024_256_cifar100_model.pth)     | 64.32     | 72.51        |
| `TinyImageNet`   | 1024 | 0.0009765  | 4.0        | [kxlkigsv_tiny_imagenet.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/kxlkigsv_0.0009765_1024_256_tiny_imagenet_model.pth)     | 42.21     | 51.84        |
| `STL-10`        | 1024 | 0.0078125  | 2.0        | [pbknx38b_stl10.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/pbknx38b_0.0078125_1024_256_stl10_model.pth)     | 87.79     | 91.70        |

#### 3.3. ResNet-50 on `ImageNet` (300 epochs)
> Setting: epochs = 300, $d$ = 8192, $\lambda_{BT}$ = 0.0051

| $\lambda_{reg}$ | Linear Acc. | Download Link to Pretrained Model | Train Log | Download Link to Linear-Probed Model | Val. Log |
| ---------- | --------------------- | ------ | ----- | ------ | ----------- |
| 0.0 (BT)   | 71.3     | [3on0l4wl_resnet50.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/3on0l4wl_0.0000_8192_1024_imagenet_resnet50.pth) | [train_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/3on0l4wl_train.txt) | [checkpoint_3tb4tcvp.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/checkpoint_3tb4tcvp.pth) | [val_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/3tb4tcvp_val.txt) |
| 0.0025     | 70.9  | [l418b9zw_resnet50.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/l418b9zw_0.0025_8192_1024_imagenet_resnet50.pth) | [train_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/l418b9zw_train.txt) | [checkpoint_09g7ytcz.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/checkpoint_09g7ytcz.pth)  | [val_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/09g7ytcz_val.txt)  |
| 0.1        | 71.6  | [13awtq23_resnet50.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/13awtq23_0.1000_8192_1024_imagenet_resnet50.pth) | [train_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/13awtq23_train.txt) | [checkpoint_pgawzr4e.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/checkpoint_pgawzr4e.pth)  | [val_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/pgawzr4e_val.txt)    |
| 1.0        | **72.2** (best) | [3fb1op86_resnet50.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/3fb1op86_1.0000_8192_1024_imagenet_resnet50.pth) | [train_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/3fb1op86_train.txt) |  [checkpoint_wvi0hle8.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/checkpoint_wvi0hle8.pth)  | [val_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/wvi0hle8_val.txt)   |
| 2.0        | 72.1 | [5n9yqio0_resnet50.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/5n9yqio0_1.0000_8192_1024_imagenet_resnet50.pth) | [train_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/5n9yqio0_train.txt) | [checkpoint_p9aeo8ga.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/checkpoint_p9aeo8ga.pth)  | [val_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/p9aeo8ga_val.txt)   |
| 3.0        | 72.0 | [q03u2xjz_resnet50.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/q03u2xjz_3.0000_8192_1024_imagenet_resnet50.pth) | [train_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/q03u2xjz_train.txt) | [checkpoint_00atvp6x.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/checkpoint_00atvp6x.pth)  | [val_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/00atvp6x_val.txt)  |

#### 3.4. ResNet-50 on `ImageNet` (1000 epochs)
> Setting: epochs = 1000, $d$ = 8192, $\lambda_{BT}$ = 0.0051, $\lambda_{reg}$=2.0

| Linear Eval. Top1 | Linear Eval. Top5 | Download Link to Pretrained Model | Train Log | Download Link to Linear-Probed Model | Val. Log |
| ----- | ----- | --------------------------------- | --------- | ------------------------------------ | -------- |
| **74.06** (best) | 91.47 | [4wpu8wmd_resnet50.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/4wpu8wmd_resnet50.pth) | [train_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/4wpu8wmd_stats.txt) | [vfd2nu64_checkpoint.pth](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/vfd2nu64_checkpoint.pth) | [val_log](https://github.com/wgcban/mix-bt/releases/download/v1.0.0/vfd2nu64_stats.txt) |

## 4 Training/Val Logs

### 3.1 Pre-trianing for 300 epochs
Logs are available on `wandb` and can access via following links:
- imagenet pre-training: https://api.wandb.ai/links/cha-yas/5olb2sar
- imagenet linear probing: https://api.wandb.ai/links/cha-yas/9tb0ksfp

Here we provide some training and validation (linear probing) statistics for Barlow Twins *vs.* Mixed Barlow Twins with `ResNet-50` backbone on `ImageNet`:

<img src="figs/in-loss-bt.png" width="256"/> <img src="figs/in-loss-reg.png" width="256"/> <img src="figs/in-linear.png" width="256"/> 

### 3.1 Pre-trianing for 1000 epochs
We also provide trianing-val statistics for our pre-trained model for 1000 epochs.
<img src="figs/in-loss-bt-1000e.png" width="256"/> <img src="figs/in-loss-reg-1000e.png" width="256"/> <img src="figs/in-linear-1000e.png" width="256"/> 

:fire: Access pre-training statistcis on wandb: [`wandb-imagenet-pretrain`](https://wandb.ai/cha-yas/Barlow-Twins-MixUp-ImageNet?workspace=user-wgcban)

## 5 Disclaimer
A large portion of the code is from [Barlow Twins HSIC](https://github.com/yaohungt/Barlow-Twins-HSIC) (for experiments on small datasets: `CIFAR-10`, `CIFAR-100`, `TinyImageNet`, and `STL-10`) and official implementation of Barlow Twins [here](https://github.com/facebookresearch/barlowtwins) (for experiments on `ImageNet`), which is a great resource for academic development.

Also, note that the implementation of SOTA methods ([SimCLR](https://arxiv.org/abs/2002.05709), [BYOL](https://arxiv.org/abs/2006.07733), and [Witening-MSE](https://arxiv.org/abs/2007.06346)) in `ssl-sota` are copied from [Witening-MSE](https://github.com/htdt/self-supervised).

We would like to thank all of them for making their repositories publicly available for the research community. 🙏

## 6 Reference
If you feel our work is useful, please consider citing our work. Thanks!
```bibtex
@misc{bandara2023guarding,
      title={Guarding Barlow Twins Against Overfitting with Mixed Samples}, 
      author={Wele Gedara Chaminda Bandara and Celso M. De Melo and Vishal M. Patel},
      year={2023},
      eprint={2312.02151},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 7 License
This code is under MIT licence, you can find the complete file [here](https://github.com/wgcban/mix-bt/blob/main/LICENSE).
