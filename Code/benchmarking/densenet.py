"""dense net in pytorch



[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""

import torch
import torch.nn as nn



#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, inner_channels_list):
        super().__init__()
        # """In  our experiments, we let each 1×1 convolution
        # produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        # """We find this design especially effective for DenseNet and
        # we refer to our network with such a bottleneck layer, i.e.,
        # to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        # as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

        r = 16
        # out_channels = in_channels
        self.expansion = 1
        # self.squeeze1 = nn.AdaptiveAvgPool2d(1)
        # self.excitation1 = nn.Sequential(
        #     nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias=False),
        #     nn.Sigmoid()
        # )
        out_channels = growth_rate
        self.squeeze2 = nn.AdaptiveAvgPool2d(1)
        self.excitation2 = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # accuracy 74% with bias
        se_input = self.bottle_neck(x)
        squeeze = self.squeeze2(se_input)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation2(squeeze)
        excitation = excitation.view(se_input.size(0), se_input.size(1), 1, 1)
        x2 = se_input * excitation.expand_as(se_input)
        x = torch.cat([x, x2], 1)
        return x

#se_block before concat
# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, growth_rate, inner_channels_list):
#         super().__init__()
#         #"""In  our experiments, we let each 1×1 convolution 
#         #produce 4k feature-maps."""
#         inner_channel = 4 * growth_rate

#         #"""We find this design especially effective for DenseNet and 
#         #we refer to our network with such a bottleneck layer, i.e., 
#         #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` , 
#         #as DenseNet-B."""
#         self.bottle_neck = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
#             nn.BatchNorm2d(inner_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
#         )
#         self.inner_channels_list = inner_channels_list.copy()
#         r = 16
#         self.expansion = 1
#         self.squeeze = []
#         self.excitation = []
#         for i in range(len(self.inner_channels_list)):
#             if i == 0:
#                 out_channels = self.inner_channels_list[0] 
#             else:
#                 out_channels = growth_rate   
#             self.squeeze = nn.AdaptiveAvgPool2d(1)
#             self.excitation.append(nn.Sequential(
#                 nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias = False),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias = False),
#                 nn.Sigmoid()
#             ))
#         self.excitation = nn.ModuleList(self.excitation)

#     def forward(self, x):
#         x_scaled = []
#         #sequential
#         for i in range(len(self.inner_channels_list)):
#             if i == 0:
#                 se_input = x[:, :self.inner_channels_list[0], :, :]
#             else:
#                 se_input = x[:, self.inner_channels_list[i-1]: self.inner_channels_list[i], :, :]
#             squeeze = self.squeeze(se_input)
#             squeeze = squeeze.view(squeeze.size(0), -1)
#             excitation = self.excitation[i](squeeze)
#             excitation = excitation.view(se_input.size(0), se_input.size(1), 1, 1)
#             x1 = se_input * excitation.expand_as(se_input)
#             x_scaled.append(x1)
#         x_scaled = torch.cat(x_scaled, 1)
#         return torch.cat([x_scaled, self.bottle_neck(x_scaled)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments 
        #consist of a batch normalization layer and an 1×1 
        #convolutional layer followed by a 2×2 average pooling 
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution 
        #with 16 (or twice the growth rate for DenseNet-BC) 
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each 
        #side of the inputs is zero-padded by one pixel to keep 
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False) 

        self.features = nn.Sequential()
        inner_channels_list = [inner_channels]
        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index], inner_channels_list))
            inner_channels += growth_rate * nblocks[index]
            
            #"""If a dense block contains m feature-maps, we let the 
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression 
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
            inner_channels_list = [inner_channels]

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1], inner_channels_list))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks, inner_channels_list):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate, inner_channels_list))
            in_channels += self.growth_rate
            inner_channels_list.append(in_channels)
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

