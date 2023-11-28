import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from transfer_datasets import TRANSFER_DATASET
import torchvision.transforms as transforms
from data_statistics import get_data_mean_and_stdev, get_data_nclass
from tqdm import tqdm

import utils

import wandb

import torchvision

def load_transform(dataset, size=32):
    mean, std = get_data_mean_and_stdev(dataset)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    return transform

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path, dataset, arch):
        super(Net, self).__init__()

        if arch=='resnet18':
            embedding_size = 512
        elif arch=='resnet50':
            embedding_size = 2048
        else:
            raise NotImplementedError

        # encoder
        from model import Model
        self.f = Model(dataset=dataset, arch=arch).f
        # classifier
        self.fc = nn.Linear(embedding_size, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}% model: {}'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100,
                                             model_path.split('/')[-1]))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Pre-trained dataset.', choices=['cifar10', 'cifar100', 'stl10', 'tiny_imagenet'])
    parser.add_argument('--transfer_dataset', default='cifar10', type=str, help='Transfer dataset (i.e., testing dataset)', choices=['cifar10', 'cifar100', 'stl-10', 'aircraft', 'cu_birds', 'dtd', 'fashionmnist', 'mnist', 'traffic_sign', 'vgg_flower'])
    parser.add_argument('--arch', default='resnet50', type=str, help='Backbone architecture for experiments', choices=['resnet50', 'resnet18'])
    parser.add_argument('--model_path', type=str, default='results/Barlow_Twins/0.005_64_128_model.pth',
                        help='The base string of the pretrained model path')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--screen', type=str, help='screen session id')
    # wandb related args
    parser.add_argument('--wandb_group', type=str, help='group for wandb')

    args = parser.parse_args()
    
    wandb.init(project=f"Barlow-Twins-MixUp-TransferLearn-[{args.dataset}-to-X]-{args.arch}", config=args, dir='/data/wbandar1/projects/ssl-aug-artifacts/wandb_logs/', group=args.wandb_group, name=f'{args.transfer_dataset}')
    run_id = wandb.run.id

    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    dataset = args.dataset
    transfer_dataset = args.transfer_dataset
    
    if dataset in ['cifar10', 'cifar100']:
        print("reshaping data into 32x32")
        resize = 32
    else:
        print("reshaping data into 64x64")
        resize = 64
        
    train_data = TRANSFER_DATASET[args.transfer_dataset](train=True, image_transforms=load_transform(args.transfer_dataset, resize))
    test_data = TRANSFER_DATASET[args.transfer_dataset](train=False, image_transforms=load_transform(args.transfer_dataset, resize))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    model = Net(num_class=get_data_nclass(args.transfer_dataset), pretrained_path=model_path, dataset=dataset, arch=args.arch).cuda()
    for param in model.f.parameters():
        param.requires_grad = False

    # optimizer with lr sheduler
    # lr_start, lr_end = 1e-2, 1e-6
    # gamma = (lr_end / lr_start) ** (1 / epochs)
    # optimizer = optim.Adam(model.fc.parameters(), lr=lr_start, weight_decay=5e-6)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # adpoted from 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.1)
    
    # optimizer with no sheuduler
    # optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    loss_criterion = nn.CrossEntropyLoss()
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

    save_name = model_path.split('.pth')[0] + '_linear.csv'

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv(save_name, index_label='epoch')
        if test_acc_1 > best_acc:
           best_acc = test_acc_1
        wandb.log(
                {
                "train_loss": train_loss,
                "train_acc@1": train_acc_1,
                "train_acc@5": train_acc_5,
                "test_loss": test_loss,
                "test_acc@1": test_acc_1,
                "test_acc@5": test_acc_5,
                "best_acc": best_acc
                }
            )
        scheduler.step()
    wandb.finish()
