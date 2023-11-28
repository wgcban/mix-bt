#!/bin/bash
is_mixup=true
batch_size=1024 #128/gpu works
lr_w=0.2 #0.2
lr_b=0.0048 #0.0048
lambda_mixup=5.0


timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"
screen -S "$session_name" -X stuff "NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_imagenet.py /data/wbandar1/datasets/imagenet1k/ --is_mixup ${is_mixup} --batch-size ${batch_size} --learning-rate-weights ${lr_w} --learning-rate-biases ${lr_b} --lambda_mixup ${lambda_mixup}^M"
screen -S "$session_name" -X detachs