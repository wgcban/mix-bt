#!/bin/bash
gpu=7
dataset=cifar10
arch=resnet18
batch_size=512

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"

# no-mixup: 0.0078125
# 1024 (x8nwefvu)
model_path=/mnt/store/wbandar1/projects/ssl-aug-artifacts/x8nwefvu/x8nwefvu_0.0078125_1024_256_cifar10_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# mixup: 0.0078125
# 1024-s4-step (lxu710q1)
model_path=/mnt/store/wbandar1/projects/ssl-aug-artifacts/lxu710q1/lxu710q1_0.0078125_1024_256_cifar10_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# 1024-s4-cosine (4wdhbpcf)
model_path=results/4wdhbpcf/4wdhbpcf_0.0078125_1024_256_cifar10_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
screen -S "$session_name" -X detachs