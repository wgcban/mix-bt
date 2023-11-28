#!/bin/bash
gpu=2
dataset=cifar10
arch=resnet50 #resnet18, resnet50
batch_size=512

timestamp=$(date +"%Y%m%d%H%M%S")
session_name="python_session_$timestamp"
echo ${session_name}
screen -dmS "$session_name"
screen -S "$session_name" -X stuff "conda activate ssl-aug^M"

# # Experiments with fixed LR
# # cifar-10, no-mixup, 0.0078125
# model_path=results/1rs9cpdk/1rs9cpdk_0.0078125_128_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 1024
# model_path=results/ndtcd1wj/ndtcd1wj_0.0078125_1024_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 2048
# model_path=results/0j0dsc2f/0j0dsc2f_0.0078125_2048_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 4096
# model_path=results/9mpwelye/9mpwelye_0.0078125_4096_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# # cifar-10, mixup, 0.0078125
# # 128
# model_path=results/71s3qho8/71s3qho8_0.0078125_128_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 1024
# model_path=results/x0bvh7zs/x0bvh7zs_0.0078125_1024_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 2048
# model_path=results/d7f1wcku/d7f1wcku_0.0078125_2048_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"
# # 4096
# model_path=results/kgzb37sa/kgzb37sa_0.0078125_4096_256_cifar10_model_1000.pth
# screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

## Experiments with cosine LR ##
# bt: 3wbdd6hb
model_path=results/3wbdd6hb/3wbdd6hb_0.0078125_1024_256_cifar10_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

# mbt: v3gwgusq
model_path=results/v3gwgusq/v3gwgusq_0.0078125_1024_256_cifar10_model.pth
screen -S "$session_name" -X stuff "CUDA_VISIBLE_DEVICES=${gpu} python linear.py --dataset ${dataset} --model_path ${model_path} --arch ${arch}^M"

screen -S "$session_name" -X detachs