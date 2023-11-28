import argparse
import os

import pandas as pd
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from thop import profile, clever_format
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

import utils
from model import Model
import math

import torchvision

import wandb

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.lr

def train(args, epoch, net, data_loader, train_optimizer):
    net.train()
    total_loss, total_loss_bt, total_loss_mix, total_num, train_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    for step, data_tuple in enumerate(train_bar, start=epoch * len(train_bar)):
        if args.lr_shed == "cosine":
            adjust_learning_rate(args, train_optimizer, data_loader, step)
        (pos_1, pos_2), _ = data_tuple
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)
        
        out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
        out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)
        c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss_bt = on_diag + lmbda * off_diag
        
        ##  MixUp (Our Contribution) ## 
        if args.is_mixup.lower() == 'true':
            index = torch.randperm(batch_size).cuda(non_blocking=True)
            alpha = np.random.beta(1.0, 1.0)
            pos_m = alpha * pos_1 + (1 - alpha) * pos_2[index, :]
            
            _, out_m = net(pos_m)
            out_m_norm = (out_m - out_m.mean(dim=0)) / out_m.std(dim=0)
            
            cc_m_1 = torch.matmul(out_m_norm.T, out_1_norm) / batch_size
            cc_m_1_gt = alpha*torch.matmul(out_1_norm.T, out_1_norm) / batch_size + \
                            (1-alpha)*torch.matmul(out_2_norm[index,:].T, out_1_norm) / batch_size

            cc_m_2 = torch.matmul(out_m_norm.T, out_2_norm) / batch_size
            cc_m_2_gt = alpha*torch.matmul(out_1_norm.T, out_2_norm) / batch_size + \
                            (1-alpha)*torch.matmul(out_2_norm[index,:].T, out_2_norm) / batch_size
            
            loss_mix = args.mixup_loss_scale*lmbda*((cc_m_1-cc_m_1_gt).pow_(2).sum() + (cc_m_2-cc_m_2_gt).pow_(2).sum())
        else:
            loss_mix = torch.zeros(1).cuda()
        ##  MixUp (Our Contribution) ##
        
        loss = loss_bt + loss_mix
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        total_loss_bt += loss_bt.item() * batch_size
        total_loss_mix += loss_mix.item() * batch_size
        
        train_bar.set_description('Train Epoch: [{}/{}] lr: {:.3f}x10-3 Loss: {:.4f} lmbda:{:.4f} bsz:{} f_dim:{} dataset: {}'.format(\
                                epoch, epochs, train_optimizer.param_groups[0]['lr'] * 1000, total_loss / total_num, lmbda, batch_size, feature_dim, dataset))
    return total_loss_bt / total_num, total_loss_mix / total_num, total_loss / total_num


def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
            (data, _), target = data_tuple
            target_bank.append(target)
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (data, _), target = data_tuple
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
    return total_top1 / total_num * 100, total_top5 / total_num * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Barlow Twins')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10, cifar100, tiny_imagenet, stl10', choices=['cifar10', 'cifar100', 'tiny_imagenet', 'stl10'])
    parser.add_argument('--arch', default='resnet50', type=str, help='Backbone architecture', choices=['resnet50', 'resnet18'])
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for embedding vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax (kNN evaluation)')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Base learning rate')
    parser.add_argument('--lr_shed', default="step", choices=["step", "cosine"], type=str, help='Learning rate scheduler: step / cosine')
    
    # for barlow twins
    parser.add_argument('--lmbda', default=0.005, type=float, help='Lambda that controls the on- and off-diagonal terms')
    parser.add_argument('--corr_neg_one', dest='corr_neg_one', action='store_true')
    parser.add_argument('--corr_zero', dest='corr_neg_one', action='store_false')
    parser.set_defaults(corr_neg_one=False)
    
    # for mixup
    parser.add_argument('--is_mixup', dest='is_mixup', type=str, default='false', choices=['true', 'false'])
    parser.add_argument('--mixup_loss_scale', dest='mixup_loss_scale', type=float, default=5.0)

    # GPU id (just for record)
    parser.add_argument('--gpu', dest='gpu', type=int, default=0)
    
    args = parser.parse_args()
    is_mixup = args.is_mixup.lower() == 'true'

    wandb.init(project=f"Barlow-Twins-MixUp-{args.dataset}-{args.arch}", config=args, dir='results/wandb_logs/')
    run_id = wandb.run.id
    dataset = args.dataset
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    lmbda = args.lmbda
    corr_neg_one = args.corr_neg_one
    
    if dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root='/data/wbandar1/datasets', train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.CIFAR10(root='/data/wbandar1/datasets', train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.CIFAR10(root='/data/wbandar1/datasets', train=False, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
    elif dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root='/data/wbandar1/datasets', train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.CIFAR100(root='/data/wbandar1/datasets', train=True, \
                                                    transform=utils.CifarPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.CIFAR100(root='/data/wbandar1/datasets', train=False, \
                                                    transform=utils.CifarPairTransform(train_transform = False), download=True)
    elif dataset == 'stl10':
        train_data = torchvision.datasets.STL10(root='/data/wbandar1/datasets', split="train+unlabeled", \
                                                    transform=utils.StlPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.STL10(root='/data/wbandar1/datasets', split="train", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.STL10(root='/data/wbandar1/datasets', split="test", \
                                                  transform=utils.StlPairTransform(train_transform = False), download=True)
    elif dataset == 'tiny_imagenet':
        # download if not exits
        if not os.path.isdir('/data/wbandar1/datasets/tiny-imagenet-200'):
            raise ValueError("First preprocess the tinyimagenet dataset...") 
            
        train_data = torchvision.datasets.ImageFolder('/data/wbandar1/datasets/tiny-imagenet-200/train', \
                                                        utils.TinyImageNetPairTransform(train_transform = True))
        memory_data = torchvision.datasets.ImageFolder('/data/wbandar1/datasets/tiny-imagenet-200/train', \
                                                      utils.TinyImageNetPairTransform(train_transform = False))
        test_data = torchvision.datasets.ImageFolder('/data/wbandar1/datasets/tiny-imagenet-200/val', \
                                                      utils.TinyImageNetPairTransform(train_transform = False))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                            drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim, dataset, args.arch).cuda()
    if dataset == 'cifar10' or dataset == 'cifar100':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    elif dataset == 'tiny_imagenet' or dataset == 'stl10':
        flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    if args.lr_shed == "step":
        m = [args.epochs - a for a in [50, 25]]
        scheduler = MultiStepLR(optimizer, milestones=m, gamma=0.2)
    c = len(memory_data.classes)

    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(run_id, lmbda, feature_dim, batch_size, dataset)
    run_id_dir = os.path.join('results/', run_id)
    if not os.path.exists(run_id_dir):
        print('Creating directory {}'.format(run_id_dir))
        os.mkdir(run_id_dir)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        loss_bt, loss_mix, train_loss = train(args, epoch, model, train_loader, optimizer)
        if args.lr_shed == "step":
            scheduler.step()
        wandb.log(
                {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]['lr'],
                "loss_bt": loss_bt,
                "loss_mix": loss_mix,
                "train_loss": train_loss}
            )
        if epoch % 5 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)

            results['train_loss'].append(train_loss)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            data_frame = pd.DataFrame(data=results, index=range(5, epoch + 1, 5))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
            wandb.log(
                {
                "test_acc@1": test_acc_1,
                "test_acc@5": test_acc_5
                }
            )
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                torch.save(model.state_dict(), 'results/{}/{}_model.pth'.format(run_id, save_name_pre))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), 'results/{}/{}_model_{}.pth'.format(run_id, save_name_pre, epoch))
    wandb.finish()
