#!/bin/bash
gpu=2
dataset=tiny_imagenet
arch=resnet50
feature_dim=4096
is_mixup=false # true, false
batch_size=256
epochs=2000
lr=0.01
lr_shed=cosine # step, cosine
mixup_loss_scale=5.0 # scale w.r.t. lambda: 0.0078125 * 5 = 0.0390625

lmbda=$(echo "scale=7; 1 / ${feature_dim}" | bc)
echo ${lmbda}

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python main.py --lmbda ${lmbda} --corr_zero --batch_size ${batch_size} --feature_dim ${feature_dim} --dataset ${dataset} --is_mixup ${is_mixup} --mixup_loss_scale ${mixup_loss_scale} --epochs ${epochs} --arch ${arch} --gpu ${gpu} --lr_shed ${lr_shed} --lr ${lr}^M"
screen -S "$session_name" -X detach