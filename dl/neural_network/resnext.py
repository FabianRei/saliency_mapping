from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch
from fnmatch import fnmatch
import math
import torch.hub
import os
import wget
from torchvision.models.resnet import ResNet, Bottleneck


def make_resnext(pretrained=False):
    # ResNext101 = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=48)
    ResNext101 = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=32)
    return ResNext101

class ResNext101(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, num_input=3):
        super().__init__()
        self.channel_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.pretrained = pretrained
        # use weights from: https://arxiv.org/pdf/1906.06423.pdf
        # model_path = '/scratch/reith/fl/experiments/models/'
        # weight_path = os.path.join(model_path, 'ResNext101_32x48d_v2.pth')
        if pretrained:
            # self.ResNext = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl', force_reload=True)
            self.ResNext = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl', force_reload=True)
            # if not os.path.exists(weight_path):
            #     wget.download(
            #         'https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNext101_32x48d_v2.pth',
            #         out=model_path)
            # self.load_model_dict(weight_path)
        else:
            self.ResNext = make_resnext()
        # print('nice')
        # models.resnet50(pretrained=pretrained)
            # models.resnet50(pretrained=pretrained)
        self.num_input = num_input
        if num_input > 3 and not pretrained:
            # experiment with higher input channels
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # standard resnet initialization for conv2d
            n = new_input.kernel_size[0] * new_input.kernel_size[1] * new_input.out_channels
            new_input.weight.data.normal_(0, math.sqrt(2. / n))
            self.ResNext.conv1 = new_input
        elif num_input > 3 and pretrained:
            input_weights = self.ResNext.conv1.weight.data
            # we divide by three, as this would yield the same activations, given the input slice was simply being
            # copied
            input_weights = input_weights.repeat(1,(num_input//3),1,1)/(num_input//3)
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            new_input.weight.data = input_weights
            self.ResNext.conv1 = new_input
            self.channel_mean = self.channel_mean.repeat(1,(num_input//3),1,1)
            self.channel_std = self.channel_std.repeat(1,(num_input//3),1,1)
        # pytorch's standard implementation throws errors at some image sizes..
        # self.ResNext.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNext.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        """
        Forward pass. Gray image is copied into pseudo 3dim rgb image and mean/std are adapted to
        the ImageNet distribution

        :param x:
        :return:
        """
        # copy to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # substract imagenet mean and scale imagenet std
        if self.pretrained:
            x -= self.channel_mean
            x /= self.channel_std
        x = self.ResNext(x)
        return F.log_softmax(x, dim=1)

    def freeze_except_fc(self):
        for name, param in self.named_parameters():
            if fnmatch(name, '*fc.*'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_model_dict(self, weight_path):
        pretrained_dict = torch.load(weight_path, map_location='cpu')['model']
        model_dict = self.ResNext.state_dict()
        for k in model_dict.keys():
            if ('module.' + k) in pretrained_dict.keys():
                model_dict[k] = pretrained_dict.get(('module.' + k))
        self.ResNext.load_state_dict(model_dict)


class ResNext101Reg(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, num_input=3):
        super().__init__()
        self.channel_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.pretrained = pretrained
        # use weights from: https://arxiv.org/pdf/1906.06423.pdf
        # model_path = '/scratch/reith/fl/experiments/models/'
        # weight_path = os.path.join(model_path, 'ResNext101_32x48d_v2.pth')
        if pretrained:
            # self.ResNext = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl', force_reload=True)
            self.ResNext = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl', force_reload=True)
            # if not os.path.exists(weight_path):
            #     wget.download(
            #         'https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNext101_32x48d_v2.pth',
            #         out=model_path)
            # self.load_model_dict(weight_path)
        else:
            self.ResNext = make_resnext()
        # print('nice')
        # models.resnet50(pretrained=pretrained)
        # we assume multiples of 3
        self.num_input = num_input
        if num_input > 3 and not pretrained:
            # experiment with higher input channels
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # standard resnet initialization for conv2d
            n = new_input.kernel_size[0] * new_input.kernel_size[1] * new_input.out_channels
            new_input.weight.data.normal_(0, math.sqrt(2. / n))
            self.ResNext.conv1 = new_input
        elif num_input > 3 and pretrained:
            input_weights = self.ResNext.conv1.weight.data
            # we divide by three, as this would yield the same activations, given the input slice was simply being
            # copied
            input_weights = input_weights.repeat(1,(num_input//3),1,1)/(num_input//3)
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            new_input.weight.data = input_weights
            self.ResNext.conv1 = new_input
            self.channel_mean = self.channel_mean.repeat(1,(num_input//3),1,1)
            self.channel_std = self.channel_std.repeat(1,(num_input//3),1,1)
        # pytorch's standard implementation throws errors at some image sizes..
        # self.ResNext.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNext.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        """
        Forward pass. Gray image is copied into pseudo 3dim rgb image and mean/std are adapted to
        the ImageNet distribution

        :param x:
        :return:
        """
        # copy to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # substract imagenet mean and scale imagenet std
        if self.pretrained:
            x -= self.channel_mean
            x /= self.channel_std
        x = self.ResNext(x)
        return x  # F.log_softmax(x, dim=1)

    def freeze_except_fc(self):
        for name, param in self.named_parameters():
            if fnmatch(name, '*fc.*'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def load_model_dict(self, weight_path):
        pretrained_dict = torch.load(weight_path, map_location='cpu')['model']
        model_dict = self.ResNext.state_dict()
        for k in model_dict.keys():
            if ('module.' + k) in pretrained_dict.keys():
                model_dict[k] = pretrained_dict.get(('module.' + k))
        self.ResNext.load_state_dict(model_dict)


if __name__ == '__main__':
    net = ResNext101(pretrained=False)
    print('nice')