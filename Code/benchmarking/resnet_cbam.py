"""senet in pytorch



[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cbam import CBAM


def single_list(x):
    """ If an Element is a single instead of a list, when a list is expected it created a single element list"""
    if x.__class__.__name__ is 'Tensor':
       return [x]
    else:
        return x

class BasicResidualSEBlock(nn.Module):
    expansion = 1
    # [global_local_attention_addition, global_attention_addition, global_local_attention_concat, global_attention_concat]
    # [global_local_attention_concat_learnable, global_local_attention_addition_learnable]
    # [standard_local_attention, identity_local_attention, pre_local_attention]
    exp_name = 'standard_cbam'
    def __init__(self, in_channels, out_channels, stride, block_num, r=16):
        super().__init__()
        if not 'concat' in self.exp_name:
            block_num = 1
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1, bias = False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias = False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.cbam = CBAM(out_channels * self.expansion * block_num, no_spatial=False, no_channel=False)

    def forward(self, x):
        if self.exp_name is 'standard_cbam':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
            else:
                current_input = x[0]    

            shortcut = self.shortcut(current_input)
            residual = self.residual(current_input)

            residual = self.cbam(residual)

            output = residual + shortcut

            return (F.relu(output), [])


class BottleneckResidualSEBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU()
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation1 = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )
        self.excitation2 = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        x = single_list(x)
        current_input = x[-1]
        shortcut = self.shortcut(current_input)

        residual = self.residual(current_input)
        new_connection = residual
        print(len(x))
        for input_ in x[: -1]:
            new_connection += input_
        squeeze1 = self.squeeze(new_connection)
        squeeze1 = squeeze1.view(squeeze1.size(0), -1)
        excitation1 = self.excitation1(squeeze1)
        excitation1 = excitation1.view(new_connection.size(0), new_connection.size(1), 1, 1)       
        squeeze2 = self.squeeze(residual)
        squeeze2 = squeeze2.view(squeeze2.size(0), -1)
        excitation2 = self.excitation2(squeeze2)
        excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
        

        output = residual * excitation1.expand_as(residual) *  excitation2.expand_as(residual) + shortcut
        x.append(F.relu(output))
        return x


class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=120):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)

        self.linear = nn.Linear(self.in_channels, class_num)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)

        x = self.stage2(x[0])
        x = self.stage3(x[0])
        x = self.stage4(x[0])
        x = F.adaptive_avg_pool2d(x[0], 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_stage(self, block, num, out_channels, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, 1))
        self.in_channels = out_channels * block.expansion

        for i in range(1, num):
            layers.append(block(self.in_channels, out_channels, 1, i + 1))

        return nn.Sequential(*layers)


def seresnet18(num_classes):
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2], class_num = num_classes)


def seresnet34(num_classes):
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3], class_num = num_classes)


def seresnet50(num_classes):
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3], class_num = num_classes)


def seresnet101(num_classes):
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 23, 3], class_num = num_classes)


def seresnet152(num_classes):
    return SEResNet(BottleneckResidualSEBlock, [3, 8, 36, 3], class_num = num_classes)
