#!/bin/bash
gpu=0
dataset=cifar10
arch=resnet18
batch_size=128
wandb_group='best-mbt'
model_path=checkpoints/4wdhbpcf_0.0078125_1024_256_cifar10_model.pth

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"
transfer_dataset='dtd'
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python evaluate_transfer.py --dataset ${dataset} --transfer_dataset ${transfer_dataset} --model_path ${model_path} --arch ${arch} --screen ${session_name} --wandb_group ${wandb_group}^M"
transfer_dataset='mnist'
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python evaluate_transfer.py --dataset ${dataset} --transfer_dataset ${transfer_dataset} --model_path ${model_path} --arch ${arch} --screen ${session_name} --wandb_group ${wandb_group}^M"
transfer_dataset='fashionmnist'
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python evaluate_transfer.py --dataset ${dataset} --transfer_dataset ${transfer_dataset} --model_path ${model_path} --arch ${arch} --screen ${session_name} --wandb_group ${wandb_group}^M"
transfer_dataset='cu_birds'
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python evaluate_transfer.py --dataset ${dataset} --transfer_dataset ${transfer_dataset} --model_path ${model_path} --arch ${arch} --screen ${session_name} --wandb_group ${wandb_group}^M"
transfer_dataset='vgg_flower'
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python evaluate_transfer.py --dataset ${dataset} --transfer_dataset ${transfer_dataset} --model_path ${model_path} --arch ${arch} --screen ${session_name} --wandb_group ${wandb_group}^M"
transfer_dataset='traffic_sign'
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python evaluate_transfer.py --dataset ${dataset} --transfer_dataset ${transfer_dataset} --model_path ${model_path} --arch ${arch} --screen ${session_name} --wandb_group ${wandb_group}^M"
transfer_dataset='aircraft'
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python evaluate_transfer.py --dataset ${dataset} --transfer_dataset ${transfer_dataset} --model_path ${model_path} --arch ${arch} --screen ${session_name} --wandb_group ${wandb_group}^M"
screen -S "$session_name" -X detach