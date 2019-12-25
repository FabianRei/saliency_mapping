from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch
from fnmatch import fnmatch
import math


class ResNet152(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, num_input=3):
        super().__init__()
        self.channel_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.ResNet = models.resnet152(pretrained=pretrained)
        self.pretrained = pretrained
        self.num_input = num_input
        if num_input > 3 and not pretrained:
            # experiment with higher input channels
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # standard resnet initialization for conv2d
            n = new_input.kernel_size[0] * new_input.kernel_size[1] * new_input.out_channels
            new_input.weight.data.normal_(0, math.sqrt(2. / n))
            self.ResNet.conv1 = new_input
        elif num_input > 3 and pretrained:
            input_weights = self.ResNet.conv1.weight.data
            # we divide by three, as this would yield the same activations, given the input slice was simply being
            # copied
            input_weights = input_weights.repeat(1,(num_input//3),1,1)/(num_input//3)
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            new_input.weight.data = input_weights
            self.ResNet.conv1 = new_input
            self.channel_mean = self.channel_mean.repeat(1,(num_input//3),1,1)
            self.channel_std = self.channel_std.repeat(1,(num_input//3),1,1)
        # pytorch's standard implementation throws errors at some image sizes..
        self.ResNet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNet.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)


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
        x = self.ResNet(x)
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


class ResNet152Reg(nn.Module):
    def __init__(self, pretrained=True, num_classes=1, num_input=3):
        super().__init__()
        self.channel_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.ResNet = models.resnet152(pretrained=pretrained)
        self.pretrained = pretrained
        # we assume multiples of 3
        self.num_input = num_input
        if num_input > 3 and not pretrained:
            # experiment with higher input channels
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # standard resnet initialization for conv2d
            n = new_input.kernel_size[0] * new_input.kernel_size[1] * new_input.out_channels
            new_input.weight.data.normal_(0, math.sqrt(2. / n))
            self.ResNet.conv1 = new_input
        elif num_input > 3 and pretrained:
            input_weights = self.ResNet.conv1.weight.data
            # we divide by three, as this would yield the same activations, given the input slice was simply being
            # copied
            input_weights = input_weights.repeat(1,(num_input//3),1,1)/(num_input//3)
            new_input = nn.Conv2d(num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            new_input.weight.data = input_weights
            self.ResNet.conv1 = new_input
            self.channel_mean = self.channel_mean.repeat(1,(num_input//3),1,1)
            self.channel_std = self.channel_std.repeat(1,(num_input//3),1,1)
        # pytorch's standard implementation throws errors at some image sizes..
        self.ResNet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNet.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)


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
        x = self.ResNet(x)
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


if __name__ == '__main__':
    net = ResNet152Reg(pretrained=True, num_input=9)
    print('db')