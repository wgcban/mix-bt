#!/bin/bash
gpu=1
dataset=cifar100
arch=resnet18
batch_size=512

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-sug^M"

# no-mixup, 0.0078125
# 1024 (428fzdbe)
model_path=/mnt/store/wbandar1/projects/ssl-aug-artifacts/428fzdbe/428fzdbe_0.0078125_1024_256_cifar100_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# mixup, 0.0078125
# 1024, step, p7d1zm50
model_path=/mnt/store/wbandar1/projects/ssl-aug-artifacts/p7d1zm50/p7d1zm50_0.0078125_1024_256_cifar100_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 1024, cosine, 2000, 76kk7scz
model_path=/data/wbandar1/projects/ssl-aug/Barlow-Twins-HSIC/Barlow-Twins-HSIC/results/76kk7scz/76kk7scz_0.0078125_1024_256_cifar100_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
screen -S "$session_name" -X detachs