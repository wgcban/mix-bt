#!/bin/bash
gpu=2
dataset=cifar100
arch=resnet50
batch_size=512

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate VideoMAE2^M"

# # cifar-100, no-mixup, 0.0078125
# # 128
# model_path=results/6xj2x2k8/6xj2x2k8_0.0078125_128_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 1024
# model_path=results/y19zvo9h/y19zvo9h_0.0078125_1024_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 2048
# model_path=results/c9b9ywrr/c9b9ywrr_0.0078125_2048_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 4096
# model_path=results/q1c1ftam/q1c1ftam_0.0078125_4096_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# # cifar-100, mixup, 0.0078125
# # 128
# model_path=results/bcuhc43l/bcuhc43l_0.0078125_128_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 1024
# model_path=results/kdux8evp/kdux8evp_0.0078125_1024_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 2048
# model_path=results/w0w2ri7g/w0w2ri7g_0.0078125_2048_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 4096
# model_path=results/jf6lt1z0/jf6lt1z0_0.0078125_4096_256_cifar100_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# # cifar-100, with cosine LR
# bt: uii3ni9z
model_path=results/uii3ni9z/uii3ni9z_0.0078125_1024_256_cifar100_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# mbt: z6ngefw7
model_path=results/z6ngefw7/z6ngefw7_0.0078125_1024_256_cifar100_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

screen -S "$session_name" -X detachs