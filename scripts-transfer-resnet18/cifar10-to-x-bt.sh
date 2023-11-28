#!/bin/bash
gpu=7
dataset=cifar10
arch=resnet18
batch_size=128

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"

# Barlow Twins model (epoch 1000)
wandb_group='bt-epoch1000'
model_path=/mnt/store/wbandar1/projects/ssl-aug-artifacts/x8nwefvu/x8nwefvu_0.0078125_1024_256_cifar10_model_1000.pth

# transfer learning
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

# Barlow Twins model (best epoch)
wandb_group='bt-bestepoch'
model_path=/mnt/store/wbandar1/projects/ssl-aug-artifacts/x8nwefvu/x8nwefvu_0.0078125_1024_256_cifar10_model.pth

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