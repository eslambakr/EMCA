import torch
import torch.nn as nn
import torch.nn.functional as F


def single_list(x):
    """ If an Element is a single instead of a list, when a list is expected it created a single element list"""
    if x.__class__.__name__ is 'Tensor':
       return [x]
    else:
        return x

class BasicResidualECABlock(nn.Module):
    expansion = 1
    # [global_local_attention_addition, global_attention_addition, global_local_attention_concat, global_attention_concat]
    # [global_local_attention_concat_learnable, global_local_attention_addition_learnable]
    # [standard_eca, global_local_attention_learnable_learnable_v1, global_local_attention_learnable_learnable_v2,
    #  global_local_attention_learnable_learnable_v3]
    # [multi_scale_global_no_exc, multi_scale_global_exc, multi_scale_global_local_v1, multi_scale_global_local_v2]
    exp_name = 'multi_scale_global_local_v2'
    def __init__(self, in_channels, out_channels, stride, block_num, bottleneck=False):
        super().__init__()
        if (not 'concat' in self.exp_name) and (not ("global_local_attention_learnable_learnable" in self.exp_name)) and (not ("multi_scale" in self.exp_name)):
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
        k_size = 3
        if self.exp_name == "multi_scale_global_no_exc":
            if block_num==1:
                self.ECA_excitation_local = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                    nn.Sigmoid()
                )
            if block_num>1:
                # Golbal only attention
                self.multi_scale_Conv1d = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=block_num, stride=block_num,
                            padding=0, bias = False),
                    nn.Sigmoid()
                )
            return
        elif self.exp_name == "multi_scale_global_exc":
            if block_num==1:
                self.ECA_excitation_local = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                    nn.Sigmoid()
                )
            if block_num>1:
                # Golbal only attention
                self.multi_scale_Conv1d = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=block_num, stride=block_num,
                            padding=0, bias = False),
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                    nn.Sigmoid()
                )
                # TODO: Add local attention and fuse them (learnable_learnable)
            return
        elif "multi_scale_global_local" in self.exp_name:
            self.ECA_excitation_local = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                nn.Sigmoid()
            )
            if block_num>1:
                self.multi_scale_Conv1d = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=1, kernel_size=block_num, stride=block_num,
                            padding=0, bias = False),
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                    nn.Sigmoid()
                )
                # Fuse local and Global attention
                if "v1" in self.exp_name:
                    self.fc = nn.Sequential(
                                nn.Linear(out_channels * self.expansion *2, out_channels * self.expansion, bias = False),
                                nn.Sigmoid()
                            )
                elif "v2" in self.exp_name:
                    self.fc = nn.Sequential(
                                nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2,
                                padding=0, bias = False),
                                nn.Sigmoid()
                            )
            return

        k_size = 3
        self.ECA_excitation_local = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )
        if 'global' in self.exp_name:
            self.ECA_excitation_global = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                nn.Sigmoid()
            )
            self.fc = nn.Sequential(
                nn.Linear(out_channels * self.expansion *2, out_channels * self.expansion, bias = False),
                nn.Sigmoid()
            )
        if 'global_local_attention_learnable_learnable' in self.exp_name:
            if "v1" in self.exp_name:
                self.in_1_1_conv = nn.Sequential(
                    nn.Conv2d(out_channels * self.expansion * block_num, out_channels * self.expansion, 1, bias = False),
                    nn.ReLU()
                )
            elif "v2" in self.exp_name:
                self.in_1_1_conv = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, stride=block_num, bias=False),
                    nn.ReLU()
                )
            elif "v3" in self.exp_name: 
                self.in_1_1_conv = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, stride=block_num, bias=False),
                    nn.ReLU()
                )
                self.fc = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, stride=2, bias=False),
                    nn.Sigmoid()
                )
            block_num = 1
            

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
        elif self.exp_name is 'standard_eca':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
            else:
                current_input = x[0]
            shortcut = self.shortcut(current_input)
            residual = self.residual(current_input)
            y = self.squeeze(residual)
            #print("1 = ", y.squeeze(-1).transpose(-1, -2).shape) # [N, C=1, L]
            excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2)) # [N, C=1, L]
            excitation = excitation.transpose(-1, -2).unsqueeze(-1) # [N, L, 1, 1]
            output = residual * excitation.expand_as(residual) + shortcut
            return (F.relu(output), [])
        elif "global_local_attention_learnable_learnable" in self.exp_name:
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2)) # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1) # [N, L, 1, 1]
                output = residual *  (0+excitation.expand_as(residual)) + shortcut
                return (F.relu(output), [residual])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim = 1)
                # learnable input
                if "v1" in self.exp_name:
                    new_connection = self.in_1_1_conv(new_connection) # [N, C, H, W]
                    y = self.squeeze(new_connection)
                    excitation1 = self.ECA_excitation_global(y.squeeze(-1).transpose(-1, -2)) # [N, C=1, L]
                elif "v2" in self.exp_name or "v3" in self.exp_name:
                    y = self.squeeze(new_connection) # [N, C, 1, 1]
                    y = y.squeeze(-1)
                    y = self.in_1_1_conv(y.permute(0,2,1)) # [N, 1, C]
                    excitation1 = self.ECA_excitation_global(y) # [N, C=1, L]
                
                y = self.squeeze(residual)
                excitation2 = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2)) # [N, C=1, L]
                local_global = torch.cat([excitation1, excitation2], dim = -1)
                excitation = self.fc(local_global) # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1) # [N, L, 1, 1]
                output = residual *  (0+excitation.expand_as(residual)) + shortcut
                x = (F.relu(output), previous_inputs)
            return x
        elif "multi_scale" in self.exp_name:
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                squeezed = y.view(y.size(0), -1)
                excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2)) # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1) # [N, L, 1, 1]
                output = residual *  (0+excitation.expand_as(residual)) + shortcut
                return (F.relu(output), [squeezed])
            else :             
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                squeezed = y.view(y.size(0), -1)  # [N, C]
                previous_inputs.append(squeezed)  # [old, new]
                # Local Attention
                if "multi_scale_global_local" in self.exp_name:
                    excitation_local = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2)) # [N, C=1, L]
                # Global Attention
                new_connection = torch.stack(previous_inputs)  # [S, N, C]
                new_connection = new_connection.permute(1,2,0).contiguous()  # [N, C, S]
                new_connection = new_connection.view(new_connection.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                new_connection = new_connection.permute(0,2,1)  # [N, 1, C*S][N, Cin, L]
                scales = self.multi_scale_Conv1d(new_connection) # [N, 1, L]
                # Fuse Local & Global
                if "multi_scale_global_local" in self.exp_name:
                    if "v1" in self.exp_name:
                        new_connection = torch.cat([excitation_local, scales], -1)  # [N, 1, C]
                        scales = self.fc(new_connection)  # [N, 1, C]
                    elif "v2" in self.exp_name:
                        new_connection = torch.stack([excitation_local, scales])  # [S, N, 1, C]
                        new_connection = new_connection.permute(0,1,3,2).squeeze(-1)  # [S, N, 1, C]
                        new_connection = new_connection.permute(1,2,0).contiguous()  # [N, C, S]
                        new_connection = new_connection.view(new_connection.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                        new_connection = new_connection.permute(0,2,1)  # [N, 1, C*S][N, Cin, L]
                        scales = self.fc(new_connection) # [N, 1, L]
                scales = scales.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * (0+scales) + shortcut
                x = (F.relu(output), previous_inputs)
            return x

class ECAResNet(nn.Module):
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


def eca_resnet18(num_classes):
    return ECAResNet(BasicResidualECABlock, [2, 2, 2, 2], class_num = num_classes)


def eca_resnet34(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 4, 6, 3], class_num = num_classes)


def eca_resnet50(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 4, 6, 3], class_num = num_classes, bottleneck=True)


def eca_resnet101(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 4, 23, 3], class_num = num_classes, bottleneck=True)


def eca_resnet152(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 8, 36, 3], class_num = num_classes, bottleneck=True)
