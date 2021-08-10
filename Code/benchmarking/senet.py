"""senet in pytorch



[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    # [global_local_attention_learnable_learnable, global_local_attention_learnable_learnable_att]
    # [standard_local_attention, identity_local_attention, pre_local_attention]
    # [multi_scale_conv1d]
    exp_name = 'global_local_attention_concat_learnable'
    def __init__(self, in_channels, out_channels, stride, block_num, r=16):
        super().__init__()
        
        if (not 'concat' in self.exp_name) and (self.exp_name != "global_local_attention_learnable_learnable") and (not "multi_scale" in self.exp_name):
            block_num = 1
        if bottleneck:
            self.expansion = 4
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
        else:      
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

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        if "multi_scale" in self.exp_name and block_num==1:
            self.excitation2 = nn.Sequential(
                nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias = False),
                nn.ReLU(),
                nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias = False),
                nn.Sigmoid()
            )

        if "multi_scale" in self.exp_name and block_num>1:
            self.multi_scale_Conv1d = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=1, kernel_size=block_num, stride=block_num,
                          padding=0, bias = False),
                nn.Sigmoid()
            )
        if "multi_scale" in self.exp_name:
            return

        self.excitation2 = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias = False),
            nn.ReLU(),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias = False),
            nn.Sigmoid()
        )
        if 'global_local_attention_learnable_learnable' == self.exp_name:
            self.in_1_1_conv = nn.Sequential(
                nn.Conv2d(out_channels * self.expansion * block_num, out_channels * self.expansion, 1, bias = False),
                nn.ReLU()
            )
            block_num = 1
        if not 'standard' in self.exp_name:
            self.excitation1 = nn.Sequential(
                nn.Linear(out_channels * self.expansion * block_num, out_channels * self.expansion // r, bias = False),
                nn.ReLU(),
                nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias = False),
                nn.Sigmoid()
            )
        
        if not 'standard' in self.exp_name:
            self.fc = nn.Sequential(
                nn.Linear(out_channels * self.expansion *2, out_channels * self.expansion, bias = False),
                nn.Sigmoid()
            )

        if 'global_local_attention_learnable_learnable_att' == self.exp_name:
            self.att = nn.MultiheadAttention(embed_dim =1, num_heads=1, kdim=64, vdim=64)

    def forward(self, x):
        
        if self.exp_name is 'global_local_attention_addition':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                output = residual *  excitation2.expand_as(residual) + shortcut
                return (F.relu(output), [residual])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                new_connection = residual
                for input_ in previous_inputs:
                    #if input_.shape[2] != residual.shape[2]:
                    #    input_ = F.adaptive_avg_pool2d(input_, residual.shape[2].item())
                    new_connection += input_
                squeeze1 = self.squeeze(new_connection)
                squeeze1 = squeeze1.view(squeeze1.size(0), -1)
                excitation1 = self.excitation1(squeeze1)
                # excitation1 = excitation1.view(new_connection.size(0), new_connection.size(1), 1, 1)       
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                # excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                
                local_global_mean = torch.mean(torch.stack([excitation1, excitation2]), 0)
                local_global_mean = local_global_mean.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * local_global_mean + shortcut
                previous_inputs.append(residual)
                x = (F.relu(output), previous_inputs)
            return x
        elif self.exp_name is 'global_local_attention_addition_learnable':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                output = residual *  excitation2.expand_as(residual) + shortcut
                return (F.relu(output), [residual])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                new_connection = residual
                for input_ in previous_inputs:
                    #if input_.shape[2] != residual.shape[2]:
                    #    input_ = F.adaptive_avg_pool2d(input_, residual.shape[2].item())
                    new_connection += input_
                squeeze1 = self.squeeze(new_connection)
                squeeze1 = squeeze1.view(squeeze1.size(0), -1)
                excitation1 = self.excitation1(squeeze1)
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                local_global = torch.cat([excitation1, excitation2], dim = 1)
                local_global = self.fc(local_global)
                local_global = local_global.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * local_global + shortcut
                previous_inputs.append(residual)
                x = (F.relu(output), previous_inputs)
            return x
        elif self.exp_name is 'global_attention_addition':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  
                output = residual + shortcut
                return (F.relu(output), [residual])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                new_connection = residual
                for input_ in previous_inputs:
                    #if input_.shape[2] != residual.shape[2]:
                    #    input_ = F.adaptive_avg_pool2d(input_, residual.shape[2].item())
                    new_connection += input_
                squeeze1 = self.squeeze(new_connection)
                squeeze1 = squeeze1.view(squeeze1.size(0), -1)
                excitation1 = self.excitation1(squeeze1)
                excitation1 = excitation1.view(new_connection.size(0), new_connection.size(1), 1, 1)       
                output = residual * excitation1.expand_as(residual) + shortcut
                previous_inputs.append(residual)
                x = (F.relu(output), previous_inputs)
            return x
        elif self.exp_name is 'global_local_attention_concat':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                output = residual *  excitation2.expand_as(residual) + shortcut
                return (F.relu(output), [residual])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim = 1)
                squeeze1 = self.squeeze(new_connection)
                squeeze1 = squeeze1.view(squeeze1.size(0), -1)
                excitation1 = self.excitation1(squeeze1)
                # excitation1 = excitation1.view(residual.size(0), residual.size(1), 1, 1)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                # excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                local_global_mean = torch.mean(torch.stack([excitation1, excitation2]), 0)
                local_global_mean = local_global_mean.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * local_global_mean + shortcut
                
                x = (F.relu(output), previous_inputs)

            return x
        elif self.exp_name is 'global_local_attention_concat_learnable':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                output = residual *  excitation2.expand_as(residual) + shortcut
                return (F.relu(output), [residual])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim = 1)
                squeeze1 = self.squeeze(new_connection)
                squeeze1 = squeeze1.view(squeeze1.size(0), -1)
                excitation1 = self.excitation1(squeeze1)
                # excitation1 = excitation1.view(residual.size(0), residual.size(1), 1, 1)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                # excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                local_global = torch.cat([excitation1, excitation2], dim = 1)
                local_global = self.fc(local_global)
                local_global = local_global.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * local_global + shortcut
                
                x = (F.relu(output), previous_inputs)

            return x
        elif self.exp_name is 'global_attention_concat':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  
                output = residual + shortcut
                return (F.relu(output), [residual])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim = 1)
                squeeze1 = self.squeeze(new_connection)
                squeeze1 = squeeze1.view(squeeze1.size(0), -1)
                excitation1 = self.excitation1(squeeze1)
                excitation1 = excitation1.view(residual.size(0), residual.size(1), 1, 1)      
                output = residual * excitation1.expand_as(residual) + shortcut
                x = (F.relu(output), previous_inputs)
            return x
        elif self.exp_name is 'multi_scale_conv1d':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * (1+excitation2.expand_as(residual)) + shortcut
                squeezed = self.squeeze(residual)
                squeezed = squeezed.view(squeezed.size(0), -1)  # [N, C]
                return (F.relu(output), [squeezed])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                squeezed = self.squeeze(residual)
                squeezed = squeezed.view(squeezed.size(0), -1)  # [N, C]
                previous_inputs.append(squeezed)  # [old, new]
                new_connection = torch.stack(previous_inputs)  # [S, N, C]
                new_connection = new_connection.permute(1,2,0).contiguous()  # [N, C, S]
                new_connection = new_connection.view(new_connection.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                new_connection = new_connection.permute(0,2,1)  # [N, 1, C*S][N, Cin, L]
                scales = self.multi_scale_Conv1d(new_connection)
                scales = scales.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * (1+scales) + shortcut
                x = (F.relu(output), previous_inputs)
            return x

        elif self.exp_name is 'standard_local_attention':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
            else:
                current_input = x[0]

            shortcut = self.shortcut(current_input)
            residual = self.residual(current_input)

            squeeze = self.squeeze(residual)
            squeeze = squeeze.view(squeeze.size(0), -1)
            excitation = self.excitation2(squeeze)
            excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

            output = residual * excitation.expand_as(residual) + shortcut

            return (F.relu(output), [])

        elif self.se_type == "identity_local_attention":
            if x.__class__.__name__ is 'Tensor':
                current_input = x
            else:
                current_input = x[0]    

            shortcut = self.shortcut(current_input)

            squeeze = self.squeeze(shortcut)
            squeeze = squeeze.view(squeeze.size(0), -1)
            excitation = self.excitation2(squeeze)
            excitation = excitation.view(shortcut.size(0), shortcut.size(1), 1, 1)

            residual = self.residual(current_input)

            output = shortcut * excitation.expand_as(shortcut) + residual

            return (F.relu(output), [])

        elif self.se_type == "pre_local_attention":
            if x.__class__.__name__ is 'Tensor':
                current_input = x
            else:
                current_input = x[0]    

            shortcut = self.shortcut(current_input)

            squeeze = self.squeeze(current_input)
            squeeze = squeeze.view(squeeze.size(0), -1)
            excitation = self.excitation2(squeeze)
            excitation = excitation.view(x.size(0), x.size(1), 1, 1)
            y = current_input * excitation.expand_as(current_input)

            residual = self.residual(y)

            output = residual + shortcut

            return (F.relu(output), [])


class SEResNet(nn.Module):
    def __init__(self, block, block_num, class_num=120, bottleneck=False):
        super().__init__()
        self.in_channels = 64
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stage1 = self._make_stage(block, block_num[0], 64, 1, bottleneck=bottleneck)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2, bottleneck=bottleneck)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2, bottleneck=bottleneck)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2, bottleneck=bottleneck)
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

    def _make_stage(self, block, num, out_channels, stride, bottleneck=False):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, 1, bottleneck=bottleneck))
        self.in_channels = out_channels * block.expansion
        for i in range(1, num):
            layers.append(block(self.in_channels, out_channels, 1, i + 1, bottleneck=bottleneck))
        return nn.Sequential(*layers)


def seresnet18(num_classes):
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2], class_num = num_classes)


def seresnet34(num_classes):
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3], class_num = num_classes)


def seresnet50(num_classes):
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3], class_num = num_classes, bottleneck=True)


def seresnet101(num_classes):
    return SEResNet(BasicResidualSEBlock, [3, 4, 23, 3], class_num = num_classes, bottleneck=True)


def seresnet152(num_classes):
    return SEResNet(BasicResidualSEBlock, [3, 8, 36, 3], class_num = num_classes, bottleneck=True)
