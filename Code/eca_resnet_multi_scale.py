import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from srm import srm_style_integration, srm_style_pooling


def single_list(x):
    """ If an Element is a single instead of a list, when a list is expected it created a single element list"""
    if x.__class__.__name__ is 'Tensor':
        return [x]
    else:
        return x

stage_att_all_last_blocks = True
class BasicResidualECABlock(nn.Module):
    expansion = 1
    # [global_local_attention_addition, global_attention_addition, global_local_attention_concat, global_attention_concat]
    # [global_local_attention_concat_learnable, global_local_attention_addition_learnable]
    # [standard_eca, global_local_attention_learnable_learnable_v1, global_local_attention_learnable_learnable_v2,
    #  global_local_attention_learnable_learnable_v3]
    # [multi_scale_global_no_exc, multi_scale_global_exc, multi_scale_global_local_v1,
    #  multi_scale_global_local_v2, multi_scale_global_local_2din_v2,
    #  multi_scale_global_local_v3]
    exp_name = ""

    def __init__(self, in_channels, out_channels, stride, block_num, bottleneck=False,
                 num_blocks_prev_stage=0, stage_id=0):
        super().__init__()
        self.block_num = block_num
        self.in_fusion_type = "ms"
        self.out_fusion_type = "ms"
        self.att_block_types = ["SE", "ECA", "SRM", "CBAM", "BAM"]
        self.att_block = self.att_block_types[1]
        self.ms_2d = False
        self.global_only = False
        # ------------------------------------------------------------------------------------------------------------
        #                                           Prev. Knowledge Attention
        # ------------------------------------------------------------------------------------------------------------
        # Note: block_att & stage_att could be activated together
        self.block_att = False  # feed output of all the blocks from the same stage only
        self.stage_att = True  # feed output of all the blocks from the prev. stage only
        # Note: stage_att_last_block is a special case of stage_att, which means u should activate stage_att to use it
        self.stage_att_last_block = False  # feed output of the block only from the prev. stage only
        self.stage_att_all_last_blocks = stage_att_all_last_blocks
        # Note: This parameter related to stage_att, if deactivate we will just repeat C to upsample the channels
        self.learnable_upsample = False

        # if (not 'concat' in self.exp_name) and (not ("global_local_attention_learnable_learnable" in self.exp_name)) and (not ("multi_scale" in self.exp_name)):
        #    block_num = 1
        if bottleneck:
            self.expansion = 4
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # Adaptive Grid ECA
        b = 1
        gamma = 2
        t = int(abs((math.log(out_channels * self.expansion, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        """
        # Adaptive Grid MSCA
        if bottleneck:
            if out_channels <= 64:
                k_size = 5
            elif out_channels <= 128:
                k_size = 5
            elif out_channels <= 256:
                k_size = 7
            elif out_channels <= 512:
                k_size = 9
        """
        #k_size = 19
        print(k_size)
        # ------------------------------------------------------------------------------------------------------------
        #                                           Local Attention
        # ------------------------------------------------------------------------------------------------------------
        if self.att_block == "ECA":
            self.ECA_excitation_local = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                nn.Sigmoid()
            )
        elif self.att_block == "SE":
            r = 16
            self.ECA_excitation_local = nn.Sequential(
                nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias=False),
                nn.ReLU(),
                nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias=False),
                nn.Sigmoid()
            )
        elif self.att_block == "SRM":
            self.squeeze = srm_style_pooling()
            self.ECA_excitation_local = srm_style_integration(out_channels * self.expansion)
        # ------------------------------------------------------------------------------------------------------------
        #                                           Global Attention
        # ------------------------------------------------------------------------------------------------------------
        if self.att_block == "ECA":
            if self.block_att:
                self.ECA_excitation_global_block = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                    nn.Sigmoid()
                )
            if self.stage_att:
                self.ECA_excitation_global_stage = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                    nn.Sigmoid()
                )
        elif self.att_block == "SE":
            self.ECA_excitation_global = nn.Sequential(
                nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias=False),
                nn.ReLU(),
                nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias=False),
                nn.Sigmoid()
            )
        elif self.att_block == "SRM":
            self.squeeze = srm_style_pooling()
            self.ECA_excitation_global = srm_style_integration(out_channels * self.expansion)
        # ------------------------------------------------------------------------------------------------------------
        #                                           Input Fusion
        # ------------------------------------------------------------------------------------------------------------
        if self.block_att:
            if self.in_fusion_type == "ms":
                if self.ms_2d:
                    # 2D
                    self.block_in_fusion = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(block_num, 1), stride=(block_num, 1),
                                  padding=(0, 0), bias=False),
                        nn.ReLU()
                    )
                else:
                    # 1D
                    self.block_in_fusion = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=block_num, stride=block_num,
                                  padding=0, bias=False),
                        nn.ReLU()
                    )
            elif self.in_fusion_type == "+":
                print("Input fusion type is +")
            elif self.in_fusion_type == "fc":
                self.block_in_fusion = nn.Sequential(
                    nn.Linear(out_channels * self.expansion * block_num, out_channels * self.expansion, bias=False),
                    nn.ReLU()
                )
        if self.stage_att:
            if self.in_fusion_type == "ms":
                if self.stage_att_last_block:
                    num_blocks_prev_stage = 1
                elif self.stage_att_all_last_blocks:
                    num_blocks_prev_stage = stage_id

                if self.learnable_upsample:
                    self.learnable_upsample_ms = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, k_size), stride=(1, 1),
                                  padding=(0, int(k_size / 2)), bias=False),
                        nn.ReLU()
                    )
                if self.ms_2d:
                    # 2D
                    self.stage_in_fusion = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=1,
                                  kernel_size=(num_blocks_prev_stage+1, 1), stride=(num_blocks_prev_stage+1, 1),
                                  padding=(0, 0), bias=False),
                        nn.ReLU()
                    )
                else:
                    # 1D
                    self.stage_in_fusion = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=1,
                                  kernel_size=num_blocks_prev_stage+1, stride=num_blocks_prev_stage+1,
                                  padding=0, bias=False),
                        nn.ReLU()
                    )
        # ------------------------------------------------------------------------------------------------------------
        #                                           Output Fusion
        # ------------------------------------------------------------------------------------------------------------
        if self.out_fusion_type == "ms":
            if self.stage_att and self.block_att:
                kernel_size = 3
            else:
                kernel_size = 2
            self.out_fusion = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=kernel_size,
                          padding=0, bias=False),
                nn.Sigmoid()
            )
        elif self.out_fusion_type == "+":
            print("Output fusion type is +")
        elif self.out_fusion_type == "fc":
            self.out_fusion = nn.Sequential(
                nn.Linear(out_channels * self.expansion * 2, out_channels * self.expansion, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.exp_name == "":
            if self.block_num == 1:
                # ---------------------------------------------------------------------------------------------
                #                                    First Block in the Stage
                # ---------------------------------------------------------------------------------------------
                current_input = x[0]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)  # [N, C, H, W]
                y = self.squeeze(residual)  # for SRM = [N, C, 2] , for others =[N, C, 1]
                if self.att_block == "ECA":
                    block_out = y.view(y.size(0), -1)  # [N, C]
                    excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                    excitation = excitation.transpose(-1, -2).unsqueeze(-1)  # [N, L, 1, 1]
                elif self.att_block == "SE":
                    squeezed = y.view(y.size(0), -1)
                    excitation = self.ECA_excitation_local(squeezed)
                    excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
                elif self.att_block == "SRM":
                    excitation = self.ECA_excitation_local(y)  # [N, C, 2]
                    squeezed = y
                output = residual * (1 + excitation.expand_as(residual)) + shortcut

                return (F.relu(output), [block_out], x[2])
            else:
                # ---------------------------------------------------------------------------------------------
                #                                    Other Blocks in the Stage
                # ---------------------------------------------------------------------------------------------
                current_input = x[0]
                block_previous_outputs = x[1]
                stage_prev_outputs = x[2]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                if self.att_block == "SRM":
                    block_previous_outputs.append(y)  # [old, new]
                else:
                    squeezed = y.view(y.size(0), -1)  # [N, C]
                    block_previous_outputs.append(squeezed)  # [old, new]
                # ---------------------------------------------------------------------------------------------
                #                                         Local Attention
                # ---------------------------------------------------------------------------------------------
                if not self.global_only:
                    if self.att_block == "ECA":
                        local_scales = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                    elif self.att_block == "SE":
                        excitation = self.ECA_excitation_local(squeezed)  # [N, C]
                        local_scales = excitation.unsqueeze(-1).permute(0, 2, 1)  # [N, C=1, L]
                    elif self.att_block == "SRM":
                        local_scales = self.ECA_excitation_local(y)
                        local_scales = local_scales.view(local_scales.shape[0], -1).unsqueeze(-1).permute(0, 2, 1)
                # ---------------------------------------------------------------------------------------------
                #                                         Input Fusion
                # ---------------------------------------------------------------------------------------------
                if self.block_att:
                    if self.in_fusion_type == "ms":
                        if self.att_block == "SRM":
                            t = torch.stack(block_previous_outputs)  # [S, N, C, 2]
                            if self.ms_2d:
                                # 2D
                                # 1
                                y = t[:, :, :, 0].unsqueeze(-1)  # [S, N, C, 1]
                                y = y.permute(1, 3, 0, 2)  # [N, 1, S, C]
                                y = self.block_in_fusion(y)  # [N, 1, 1, C]
                                y1 = torch.squeeze(y, dim=1)  # [N, 1, C]
                                # 2
                                y = t[:, :, :, 1].unsqueeze(-1)  # [S, N, C, 1]
                                y = y.permute(1, 3, 0, 2)  # [N, 1, S, C]
                                y = self.block_in_fusion(y)  # [N, 1, 1, C]
                                y2 = torch.squeeze(y, dim=1)  # [N, 1, C]
                                y = torch.stack([y1.view(y1.size(0), -1), y2.view(y2.size(0), -1)])  # [S, N, C]
                                y = y.permute(1, 2, 0)
                            else:
                                # 1D
                                # 1
                                y = t[:, :, :, 0]
                                y = y.permute(1, 2, 0).contiguous()  # [N, C, S]
                                y = y.view(y.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                                y = y.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                                y1 = self.block_in_fusion(y)  # [N, 1, L]
                                # 2
                                y = t[:, :, :, 1]
                                y = y.permute(1, 2, 0).contiguous()  # [N, C, S]
                                y = y.view(y.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                                y = y.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                                y2 = self.block_in_fusion(y)  # [N, 1, L]
                                y = torch.stack([y1.view(y1.size(0), -1), y2.view(y2.size(0), -1)])  # [S, N, C]
                                y = y.permute(1, 2, 0)
                        else:
                            if self.ms_2d:
                                # 2D
                                y = torch.stack(block_previous_outputs).unsqueeze(-1)  # [S, N, C, 1]
                                y = y.permute(1, 3, 0, 2)  # [N, 1, S, C]
                                y = self.block_in_fusion(y)  # [N, 1, 1, C]
                                y_block = torch.squeeze(y, dim=1)  # [N, 1, C]
                            else:
                                # 1D
                                y = torch.stack(block_previous_outputs)  # [S, N, C]
                                y = y.permute(1, 2, 0).contiguous()  # [N, C, S]
                                y = y.view(y.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                                y = y.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                                y_block = self.block_in_fusion(y)  # [N, 1, L]
                    elif self.in_fusion_type == "+":
                        y = torch.sum(torch.stack(block_previous_outputs, dim=0), dim=0)  # [N, C]
                        y = y.unsqueeze(-1).permute(0, 2, 1)  # [N, 1, L]
                    elif self.in_fusion_type == "fc":
                        y = torch.cat(block_previous_outputs, dim=-1)  # [N, C]
                        y = y.unsqueeze(-1).permute(0, 2, 1)  # [N, 1, L]
                        y = self.block_in_fusion(y)  # [N, 1, L]
                if self.stage_att:
                    if self.in_fusion_type == "ms":
                        if self.att_block == "SRM":
                            t = torch.stack(stage_prev_outputs)  # [S, N, C, 2]
                            if self.ms_2d:
                                # 2D
                                # 1
                                y = t[:, :, :, 0].unsqueeze(-1)  # [S, N, C, 1]
                                y = y.permute(1, 3, 0, 2)  # [N, 1, S, C]
                                y = self.stage_in_fusion(y)  # [N, 1, 1, C]
                                y1 = torch.squeeze(y, dim=1)  # [N, 1, C]
                                # 2
                                y = t[:, :, :, 1].unsqueeze(-1)  # [S, N, C, 1]
                                y = y.permute(1, 3, 0, 2)  # [N, 1, S, C]
                                y = self.stage_in_fusion(y)  # [N, 1, 1, C]
                                y2 = torch.squeeze(y, dim=1)  # [N, 1, C]
                                y = torch.stack([y1.view(y1.size(0), -1), y2.view(y2.size(0), -1)])  # [S, N, C]
                                y = y.permute(1, 2, 0)
                            else:
                                # 1D
                                # 1
                                y = t[:, :, :, 0]
                                y = y.permute(1, 2, 0).contiguous()  # [N, C, S]
                                y = y.view(y.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                                y = y.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                                y1 = self.stage_in_fusion(y)  # [N, 1, L]
                                # 2
                                y = t[:, :, :, 1]
                                y = y.permute(1, 2, 0).contiguous()  # [N, C, S]
                                y = y.view(y.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                                y = y.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                                y2 = self.stage_in_fusion(y)  # [N, 1, L]
                                y = torch.stack([y1.view(y1.size(0), -1), y2.view(y2.size(0), -1)])  # [S, N, C]
                                y = y.permute(1, 2, 0)
                        else:
                            if self.stage_att_last_block:
                                y = stage_prev_outputs[-1].unsqueeze(0)  # [S, N, C] s=1
                            if self.ms_2d:
                                # 2D
                                if not self.stage_att_last_block:
                                    y = torch.stack(stage_prev_outputs)  # [S, N, C]
                                # Extend the previous scale feature map across C dim. using learnable 1D-Conv:
                                if residual.shape[1] != y.size(2):
                                    if self.learnable_upsample:
                                        y = y.permute(1, 0, 2).unsqueeze(1)  # [N, 1, S, C]
                                        y = self.learnable_upsample_ms(y)  # [N, 2, S, C]
                                        y = y.permute(0,2,1,3).contiguous()  # [N, S, 2, C]
                                        y = y.view(y.shape[0], y.shape[1], -1)  # [N, S, 2*C]
                                        y = y.permute(1, 0, 2)  # [S, N, 2*C]
                                    else:
                                        # Repeat the previous scale feature map across C dim.:
                                        y = y.repeat([1, 1, int(residual.shape[1] / y.size(2))])
                                # Concatenate the previous scale feature map and the current output residual block:
                                y = torch.cat((y, squeezed.unsqueeze(0)), dim=0)
                                y = y.unsqueeze(-1)  # [S, N, C, 1]

                                y = y.permute(1, 3, 0, 2)  # [N, 1, S, C]
                                y = self.stage_in_fusion(y)  # [N, 1, 1, C]
                                y_stage = torch.squeeze(y, dim=1)  # [N, 1, C]
                            else:
                                # 1D
                                if not self.stage_att_last_block:
                                    y = torch.stack(stage_prev_outputs)  # [S, N, C]
                                # Extend the previous scale feature map across C dim. using learnable 1D-Conv:
                                if residual.shape[1] != y.size(2):
                                    if self.learnable_upsample:
                                        y = y.permute(1, 0, 2).unsqueeze(1)  # [N, 1, S, C]
                                        y = self.learnable_upsample_ms(y)  # [N, 2, S, C]
                                        y = y.permute(0, 2, 1, 3)  # [N, S, 2, C]
                                        y = y.view(y.shape[0], y.shape[1], -1)  # [N, S, 2*C]
                                        y = y.permute(1, 0, 2)  # [S, N, 2*C]
                                    else:
                                        # Repeat the previous scale feature map across C dim.:
                                        y = y.repeat([1, 1, int(residual.shape[1] / y.size(2))])

                                # Concatenate the previous scale feature map and the current output residual block:
                                y = torch.cat((y, squeezed.unsqueeze(0)), dim=0)
                                y = y.permute(1, 2, 0).contiguous()  # [N, C, S]
                                y = y.view(y.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                                y = y.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                                y_stage = self.stage_in_fusion(y)  # [N, 1, L]
                    elif self.in_fusion_type == "+":
                        y = torch.sum(torch.stack(stage_prev_outputs, dim=0), dim=0)  # [N, C]
                        y = y.unsqueeze(-1).permute(0, 2, 1)  # [N, 1, L]
                    elif self.in_fusion_type == "fc":
                        y = torch.cat(stage_prev_outputs, dim=-1)  # [N, C]
                        y = y.unsqueeze(-1).permute(0, 2, 1)  # [N, 1, L]
                        y = self.stage_in_fusion(y)  # [N, 1, L]
                # ---------------------------------------------------------------------------------------------
                #                                         Global Attention
                # ---------------------------------------------------------------------------------------------
                if self.att_block == "ECA" or self.att_block == "SE" or self.att_block == "SRM":
                    if self.stage_att:
                        global_scales_stage = self.ECA_excitation_global_stage(y_stage)  # [N, 1, L]
                    if self.block_att:
                        global_scales_block = self.ECA_excitation_global_block(y_block)  # [N, 1, L]
                if self.att_block == "SRM":
                    global_scales = self.ECA_excitation_global(y)  # [N, 1, L]
                    global_scales = global_scales.view(global_scales.shape[0], -1).unsqueeze(-1).permute(0, 2, 1)
                # ---------------------------------------------------------------------------------------------
                #                                         Output Fusion
                # ---------------------------------------------------------------------------------------------
                if self.global_only:
                    scales = global_scales
                else:
                    if self.out_fusion_type == "ms":
                        if self.stage_att and self.block_att:
                            y = torch.stack([local_scales, global_scales_stage, global_scales_block])  # [S, N, 1, C]
                        elif self.stage_att:
                            y = torch.stack([local_scales, global_scales_stage])  # [S, N, 1, C]
                        elif self.block_att:
                            y = torch.stack([local_scales, global_scales_block])  # [S, N, 1, C]

                        y = y.permute(0, 1, 3, 2).squeeze(-1)  # [S, N, 1, C]
                        y = y.permute(1, 2, 0).contiguous()  # [N, C, S]
                        y = y.view(y.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                        y = y.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                        scales = self.out_fusion(y)  # [N, 1, L]
                    elif self.out_fusion_type == "+":
                        scales = local_scales + global_scales
                    elif self.out_fusion_type == "fc":
                        y = torch.cat([local_scales, global_scales], -1)  # [N, 1, C]
                        scales = self.out_fusion(y)
                scales = scales.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * (1 + scales) + shortcut
                x = (F.relu(output), block_previous_outputs, stage_prev_outputs)
            return x
        elif self.exp_name is 'global_local_attention_addition':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                squeezed = y.squeeze(-1).permute(0, 2, 1)  # [N, 1, C]
                excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1)  # [N, L, 1, 1]
                output = residual * (1 + excitation.expand_as(residual)) + shortcut
                return (F.relu(output), [squeezed])
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                squeezed = y.squeeze(-1).permute(0, 2, 1)  # [N, 1, C]
                # Local Attention
                excitation_local = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                # Global Attention
                previous_inputs.append(squeezed)  # [old, new]
                new_connection = torch.sum(torch.stack(previous_inputs, dim=0), dim=0)
                excitation_global = self.ECA_excitation_global(new_connection)
                # Fuse Local & Global Attentions
                local_global_mean = excitation_local + excitation_global
                local_global_mean = local_global_mean.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * (1 + local_global_mean) + shortcut

                x = (F.relu(output), previous_inputs)
            return x
        elif self.exp_name is 'global_local_attention_addition_learnable':
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                squeezed = y.view(y.size(0), -1)
                excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1)  # [N, L, 1, 1]
                output = residual * (1 + excitation.expand_as(residual)) + shortcut
                return (F.relu(output), [squeezed])
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                new_connection = residual
                for input_ in previous_inputs:
                    new_connection += input_
                # Global Attention
                y = self.squeeze(new_connection)
                squeeze1 = y.view(y.size(0), -1)
                excitation1 = self.ECA_excitation_global(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                # Local Attention
                y = self.squeeze(residual)
                squeeze2 = y.view(y.size(0), -1)
                excitation2 = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                # Fuse Local & Global
                local_global = torch.cat([excitation1, excitation2], dim=-1)
                local_global = self.fc(local_global)
                local_global = local_global.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * (1 + local_global) + shortcut
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
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                new_connection = residual
                for input_ in previous_inputs:
                    # if input_.shape[2] != residual.shape[2]:
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
                output = residual * excitation2.expand_as(residual) + shortcut
                return (F.relu(output), [residual])
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim=1)
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
                output = residual * excitation2.expand_as(residual) + shortcut
                return (F.relu(output), [residual])
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim=1)
                squeeze1 = self.squeeze(new_connection)
                squeeze1 = squeeze1.view(squeeze1.size(0), -1)
                excitation1 = self.excitation1(squeeze1)
                # excitation1 = excitation1.view(residual.size(0), residual.size(1), 1, 1)  
                squeeze2 = self.squeeze(residual)
                squeeze2 = squeeze2.view(squeeze2.size(0), -1)
                excitation2 = self.excitation2(squeeze2)
                # excitation2 = excitation2.view(residual.size(0), residual.size(1), 1, 1)
                local_global = torch.cat([excitation1, excitation2], dim=1)
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
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim=1)
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
            # print("1 = ", y.squeeze(-1).transpose(-1, -2).shape) # [N, C=1, L]
            excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
            excitation = excitation.transpose(-1, -2).unsqueeze(-1)  # [N, L, 1, 1]
            output = residual * excitation.expand_as(residual) + shortcut
            return (F.relu(output), [])
        elif "global_local_attention_learnable_learnable" in self.exp_name:
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1)  # [N, L, 1, 1]
                output = residual * (0 + excitation.expand_as(residual)) + shortcut
                return (F.relu(output), [residual])
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                previous_inputs.append(residual)
                new_connection = torch.cat(previous_inputs, dim=1)
                # learnable input
                if "v1" in self.exp_name:
                    new_connection = self.in_1_1_conv(new_connection)  # [N, C, H, W]
                    y = self.squeeze(new_connection)
                    excitation1 = self.ECA_excitation_global(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                elif "v2" in self.exp_name or "v3" in self.exp_name:
                    y = self.squeeze(new_connection)  # [N, C, 1, 1]
                    y = y.squeeze(-1)
                    y = self.in_1_1_conv(y.permute(0, 2, 1))  # [N, 1, C]
                    excitation1 = self.ECA_excitation_global(y)  # [N, C=1, L]

                y = self.squeeze(residual)
                excitation2 = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                local_global = torch.cat([excitation1, excitation2], dim=-1)
                excitation = self.fc(local_global)  # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1)  # [N, L, 1, 1]
                output = residual * (0 + excitation.expand_as(residual)) + shortcut
                x = (F.relu(output), previous_inputs)
            return x
        elif "multi_scale" in self.exp_name:
            if x.__class__.__name__ is 'Tensor':
                current_input = x
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                squeezed = y.view(y.size(0), -1)
                excitation = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                excitation = excitation.transpose(-1, -2).unsqueeze(-1)  # [N, L, 1, 1]
                output = residual * (1 + excitation.expand_as(residual)) + shortcut
                return (F.relu(output), [squeezed])
            else:
                current_input = x[0]
                previous_inputs = x[1]
                shortcut = self.shortcut(current_input)
                residual = self.residual(current_input)
                y = self.squeeze(residual)
                squeezed = y.view(y.size(0), -1)  # [N, C]
                previous_inputs.append(squeezed)  # [old, new]
                # Local Attention
                if "multi_scale_global_local" in self.exp_name:
                    excitation_local = self.ECA_excitation_local(y.squeeze(-1).transpose(-1, -2))  # [N, C=1, L]
                # Global Attention
                if "2din" in self.exp_name:
                    new_connection = torch.stack(previous_inputs).unsqueeze(-1)  # [S, N, C, 1]
                    new_connection = new_connection.permute(1, 3, 0, 2)  # [N, 1, S, C]
                    scales = self.multi_scale_Conv2d(new_connection)  # [N, 1, 1, C]
                    scales = torch.squeeze(scales, dim=1)  # [N, 1, C]
                else:
                    """
                    new_connection = torch.stack(previous_inputs).unsqueeze(-1)  # [S, N, C, 1]
                    new_connection = new_connection.permute(1,3,0,2)  # [N, 1, S, C]
                    scales = self.multi_scale_Conv1d_reimp(new_connection)  # [N, 1, 1, C]
                    scales = torch.squeeze(scales, dim=1)  # [N, 1, C]
                    """
                    new_connection = torch.stack(previous_inputs)  # [S, N, C]
                    new_connection = new_connection.permute(1, 2, 0).contiguous()  # [N, C, S]
                    new_connection = new_connection.view(new_connection.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                    new_connection = new_connection.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                    scales = self.multi_scale_Conv1d(new_connection)  # [N, 1, L]

                # Fuse Local & Global
                if "multi_scale_global_local" in self.exp_name:
                    if "v1" in self.exp_name:
                        new_connection = torch.cat([excitation_local, scales], -1)  # [N, 1, C]
                        scales = self.fc(new_connection)  # [N, 1, C]
                    elif "v2" in self.exp_name:
                        new_connection = torch.stack([excitation_local, scales])  # [S, N, 1, C]
                        new_connection = new_connection.permute(0, 1, 3, 2).squeeze(-1)  # [S, N, 1, C]
                        new_connection = new_connection.permute(1, 2, 0).contiguous()  # [N, C, S]
                        new_connection = new_connection.view(new_connection.shape[0], -1).unsqueeze(-1)  # [N, C*S, 1]
                        new_connection = new_connection.permute(0, 2, 1)  # [N, 1, C*S][N, Cin, L]
                        scales = self.fc(new_connection)  # [N, 1, L]
                    elif "v3" in self.exp_name:
                        scales = scales + excitation_local
                scales = scales.view(residual.size(0), residual.size(1), 1, 1)
                output = residual * (1 + scales) + shortcut
                x = (F.relu(output), previous_inputs)
            return x


class ECAResNet(nn.Module):
    def __init__(self, block, block_num, class_num=120, bottleneck=False):
        super().__init__()
        if bottleneck:
            self.extension = 4
        else:
            self.extension = 1

        self.in_channels = 64

        """
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        """
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False))
		
			
        self.stage1 = self._make_stage(block, block_num[0], 64, 1, bottleneck=bottleneck,
                                       num_blocks_prev_stage=1, stage_id=1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2, bottleneck=bottleneck,
                                       num_blocks_prev_stage=block_num[0], stage_id=2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2, bottleneck=bottleneck,
                                       num_blocks_prev_stage=block_num[1], stage_id=3)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2, bottleneck=bottleneck,
                                       num_blocks_prev_stage=block_num[2], stage_id=4)
        self.linear = nn.Linear(self.in_channels, class_num)
        self.squeeze = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if stage_att_all_last_blocks:
            x = self.pre(x)
            last_block_0 = self.squeeze(x).view(x.size(0), -1)  # [N, C]
            last_block_in_all_stages = [last_block_0.repeat([1, self.extension])]

            x = self.stage1((x, None, last_block_in_all_stages))
            last_block_1 = x[1][-1]  # [N, C]
            last_block_in_all_stages = [last_block_0.repeat([1, 2*self.extension]), last_block_1.repeat([1, 2])]

            x = self.stage2((x[0], None, last_block_in_all_stages))
            last_block_2 = x[1][-1]  # [N, C]
            last_block_in_all_stages = [last_block_0.repeat([1, 4*self.extension]), last_block_1.repeat([1, 4]),
                                        last_block_2.repeat([1, 2])]

            x = self.stage3((x[0], None, last_block_in_all_stages))
            last_block_3 = x[1][-1]  # [N, C]
            last_block_in_all_stages = [last_block_0.repeat([1, 8*self.extension]), last_block_1.repeat([1, 8]),
                                        last_block_2.repeat([1, 4]), last_block_3.repeat([1, 2])]

            x = self.stage4((x[0], None, last_block_in_all_stages))
        else:
            x = self.pre(x)
            x = self.stage1((x, None, [self.squeeze(x).view(x.size(0), -1)]))
            x = self.stage2((x[0], None, x[1]))
            x = self.stage3((x[0], None, x[1]))
            x = self.stage4((x[0], None, x[1]))
        x = F.adaptive_avg_pool2d(x[0], 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_stage(self, block, num, out_channels, stride, bottleneck=False, num_blocks_prev_stage=0, stage_id=0):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, 1, bottleneck=bottleneck,
                            num_blocks_prev_stage=num_blocks_prev_stage, stage_id=stage_id))
        if bottleneck:
            self.in_channels = out_channels * 4
        else:
            self.in_channels = out_channels * 1
        for i in range(1, num):
            layers.append(block(self.in_channels, out_channels, 1, i + 1, bottleneck=bottleneck,
                                num_blocks_prev_stage=num_blocks_prev_stage, stage_id=stage_id))
        return nn.Sequential(*layers)


def eca_resnet18(num_classes):
    return ECAResNet(BasicResidualECABlock, [2, 2, 2, 2], class_num=num_classes)


def eca_resnet34(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 4, 6, 3], class_num=num_classes)


def eca_resnet50(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 4, 6, 3], class_num=num_classes, bottleneck=True)


def eca_resnet101(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 4, 23, 3], class_num=num_classes, bottleneck=True)


def eca_resnet152(num_classes):
    return ECAResNet(BasicResidualECABlock, [3, 8, 36, 3], class_num=num_classes, bottleneck=True)
