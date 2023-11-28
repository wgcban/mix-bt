import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10', arch='resnet50'):
        super(Model, self).__init__()

        self.f = []
        if arch == 'resnet18':
            temp_model = resnet18().named_children()
            embedding_size = 512
        elif arch == 'resnet50':
            temp_model = resnet50().named_children()
            embedding_size = 2048
        else:
            raise NotImplementedError
        
        for name, module in temp_model:
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10' or dataset == 'cifar100':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(embedding_size, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
