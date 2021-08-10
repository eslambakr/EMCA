import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from conf import settings
import copy

data_img_size = settings.IMG_SIZE


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def single_list(x):
    """ If an Element is a single instead of a list, when a list is expected it created a single element list"""
    if x.__class__.__name__ is 'Tensor':
       return [x]
    else:
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask= None,
                    src_key_padding_mask = None,
                    pos = None):
        src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class BasicResidualSEBlock(nn.Module):
    expansion = 1
    # [global_local_attention_addition, global_attention_addition, global_local_attention_concat, global_attention_concat]
    # [global_local_attention_concat_learnable, global_local_attention_addition_learnable]
    # [self_local_spatial_ch_direct_att, self_local_spatial_ch_seperate_att]
    # [self_local_spatial_att, self_local_ch_att]
    def __init__(self, in_channels, out_channels, stride, block_num, r=16, stage_num=0):
        global data_img_size
        super().__init__()
        self.stage_num = stage_num
        self.exp_name = 'self_local_ch_att_simple'
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

        if stage_num > 0:  # Add attention after first stage, to avoid big feature maps.
            if self.exp_name == 'self_local_spatial_att':
                d_model = out_channels*self.expansion*block_num
            elif self.exp_name == 'self_local_ch_att_simple':
                self.squeeze_1d = nn.AdaptiveAvgPool1d(1)
                
                self.self_att = []
                for i in range(3):
                    self.self_att.append(nn.MultiheadAttention(embed_dim =1, num_heads=1))
                self.self_att = nn.ModuleList(self.self_att)
                #self.fuse_self_att = nn.MultiheadAttention(embed_dim =1, num_heads=1)
                self.relu = _get_activation_fn(activation="relu")

            elif self.exp_name == 'self_local_ch_att':
                #self.d_model = int(data_img_size / (2**(stage_num-1)))**2
                
                if settings.IMG_SIZE == 32 and False:
                    if stage_num == 4:
                        d_model = int(data_img_size / (8))**2
                    elif stage_num == 3:
                        d_model = int(data_img_size / (4))**2
                    elif stage_num == 1 or stage_num == 2:
                        d_model = int(data_img_size / (2))**2
                elif settings.IMG_SIZE == 64 and False:
                    if stage_num == 4:
                        d_model = int(data_img_size / (8))**2
                    elif stage_num == 1 or stage_num == 2 or stage_num == 3:
                        d_model = int(data_img_size / (4))**2
                
                self.d_model = 64
                nhead = 4
                dim_feedforward = 256
                dropout = 0.1
                num_layers = 3
                #self.c_shrink = 128
                
                self.hw_TO_dmodel_proj = nn.Linear(int(data_img_size / (2**(stage_num-1)))**2, self.d_model, bias = False)
                self.shrink_C = nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias = False)
                self.extend_C = nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias = False)
                # Squeeze
                self.dmodel_TO_1_proj = nn.Linear(self.d_model, 1, bias = False)
                #self.squeeze_1d = nn.AdaptiveAvgPool1d(1)
            self.eps = 1e-6
            self.scale = 2 * math.pi
            self.temperature = 10000

            """
            self.excitation = nn.Sequential(
                                nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias = False),
                                nn.ReLU(),
                                nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias = False),
                                nn.Sigmoid()
                            )
            """
            # SE Block
            """
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation2 = nn.Sequential(
                                nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r, bias = False),
                                nn.ReLU(),
                                nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion, bias = False),
                                nn.Sigmoid()
                            ) 
            """  
            if self.exp_name == 'self_local_spatial_att' or self.exp_name == 'self_local_ch_att':
                # creating Self-Attention:
                encoder_layer = TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, dropout)
                self.layers = _get_clones(encoder_layer, num_layers)
                
                """
                self.self_att = nn.MultiheadAttention(embed_dim =self.d_model, num_heads=nhead, dropout=dropout)
                self.linear1 = nn.Linear(self.d_model, self.d_model // r, bias = False)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(self.d_model // r, self.d_model, bias = False)

                self.norm1 = nn.LayerNorm(self.d_model)
                self.norm2 = nn.LayerNorm(self.d_model)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                self.activation = _get_activation_fn(activation="relu")
                """

                """
                self.fc = nn.Sequential(
                                            nn.Linear(out_channels * self.expansion *2, out_channels * self.expansion, bias = False),
                                            nn.Sigmoid()
                                        )
                """
        
        self.sigmoid_activation = nn.Sigmoid()

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
        elif 'self_local' in self.exp_name:
            if x.__class__.__name__ is 'Tensor':
                current_input = x
            else:
                current_input = x[0]

            shortcut = self.shortcut(current_input)
            residual = self.residual(current_input)  # residual shape is [N, C, H, W]

            if self.stage_num > 0:
                # Add attention after first stage, to avoid big feature maps.
                #src = residual.detach().clone()
                bs, c, h, w = residual.shape  # [N,C,H,W]
                if self.exp_name == 'self_local_spatial_att':
                    if h > 16:
                        # Downsample --> Attention --> Upsampling to avoid big feature maps.
                        src = F.interpolate(src, (16, 16), mode='bilinear')
                    src = src.flatten(2).permute(2, 0, 1)  # [H*W, N, C]
                    # TODO: Add pos encoding
                    #pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W, N, C]
                    q = k = src
                    src = self.self_att(q, k, value=src)[0]
                    src = self.activation(self.norm1(src))
                    #print("stage_num = ", self.stage_num)
                    #print("q = ", q.shape)
                    
                    if h > 16:
                        # Downsample --> Attention --> Upsampling to avoid big feature maps.
                        excitation = src.permute(1,2,0).reshape((bs, c, 16, 16))
                        excitation = F.interpolate(excitation, (h, w), mode='bilinear')
                    else:
                        excitation =src.permute(1,2,0).reshape((bs, c, h, w))
                elif self.exp_name == 'self_local_ch_att_simple':
                    src = residual.flatten(2)  # [N, C, H*W]
                    src = self.squeeze_1d(src)
                    inter_src = []
                    for i in range(3):
                        q = k = src
                        src = self.self_att[i](q, k, value=src)[0]
                        src = self.relu(src)
                        inter_src.append(src)
                    
                    #src2 = torch.cat(inter_src, dim=1)
                    #src = self.fuse_self_att(src, src2, value=src2)[0]
                    src = self.sigmoid_activation(src)
                    src = src.view(bs, c, 1, 1)
                    excitation = src.expand_as(residual)
                elif self.exp_name == 'self_local_ch_att':
                    #print("src before= ", src.shape)
                    src = residual.flatten(2)  # [N, C, H*W]
                    src = self.hw_TO_dmodel_proj(src)  # [N, C, d]
                    src = self.shrink_C(src.permute(0, 2, 1))  # [N, d, C/16]
                    src = src.permute(2, 0, 1)  # [C/16, N, d]
                    # TODO: Add pos encoding
                    channel_embed = torch.ones(size=(src.size(1), src.size(0)), dtype=torch.float32, device=src.device)
                    channel_embed = channel_embed.cumsum(1, dtype=torch.float32)
                    channel_embed = channel_embed / (channel_embed[:, -1:] + self.eps) * self.scale
                    dim_t = torch.arange(self.d_model, dtype=torch.float32, device=src.device)
                    dim_t = self.temperature ** (2 * (dim_t // 2) / self.d_model)
                    pos_ch = channel_embed[:, :, None] / dim_t
                    pos_ch = torch.stack((pos_ch[:, :, 0::2].sin(), pos_ch[:, :, 1::2].cos()), dim=3).flatten(2).permute(1,0,2)  

                    output = src
                    for layer in self.layers:
                        output = layer(output, pos=pos_ch)
                    src = self.extend_C(output.permute(1, 2, 0))  # [N, d, C]
                    """
                    q = k = src + pos_ch
                    src2 = self.self_att(q, k, value=src)[0]
                    src = src + self.dropout1(src2)  # [C, N, d]
                    src = self.norm1(src)
                    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
                    src = src + self.dropout2(src2)
                    src = self.norm2(src)
                    src = self.activation(self.norm1(src))
                    """
                    #src = self.activation(src2)
                    #src = self.extend_C(src.permute(1, 2, 0))  # [N, d, C]
                    #src = src.permute(1, 0, 2)

                    #src = self.squeeze_1d(src.permute(1, 0, 2))[:, :, 0]  # [N, C]
                    src = self.dmodel_TO_1_proj(src.permute(0, 2, 1))[:, :, 0]  # [N, C]
                    src = self.sigmoid_activation(src)
                    #src = self.excitation(src)
                    src = src.view(bs, c, 1, 1)
                    excitation = src.expand_as(residual)
                    
                    # SE
                    """
                    squeeze = self.squeeze(residual)
                    squeeze = squeeze.view(squeeze.size(0), -1)
                    excitation2 = self.excitation2(squeeze)
                    #excitation2 = excitation.view(src.size(0), src.size(1), 1, 1)
                    # Fusing
                    excitation = torch.cat([src, excitation2], dim = 1)
                    excitation = self.fc(excitation)
                    excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
                    excitation = excitation.expand_as(residual)
                    """

                elif self.exp_name == 'self_local_spatial_att_wrong':
                    src1 = src.flatten(2).permute(2, 0, 1)  # [H*W, N, C]
                    src = torch.mean(src1,2).unsqueeze(2)  # [H*W, N, 1]
                    # Add pos encoding
                    pos_enc = torch.ones(size=(h*w, bs, 1), dtype=torch.float32, device=src.device)
                    pos_enc = pos_enc.cumsum(0, dtype=torch.float32)  # [H*W, N, 1]
                    """
                    y_embed = pos_enc.cumsum(1, dtype=torch.float32)
                    x_embed = pos_enc.cumsum(2, dtype=torch.float32)
                    y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
                    x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
                    num_pos_feats = int(c/2)
                    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=src.device)
                    dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)
                    pos_x = x_embed[:, :, :, None] / dim_t
                    pos_y = y_embed[:, :, :, None] / dim_t
                    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
                    pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [N,C,H,W]
                    pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [H*W, N, C]
                    """
                    q = k = src + pos_enc
                    src = self.self_att(q, k, value=src)[0]
                    src = self.activation(src)
                    src = src.expand_as(src1)  # [H*W, N, C]
                    excitation =src.permute(1,2,0).reshape((bs, c, h, w))
                elif self.exp_name == 'self_local_ch_att_wrong':
                    src1 = src.flatten(2).permute(1, 0, 2)  # [C, N, H*W]
                    src = torch.mean(src1,2).unsqueeze(2)  # [C, N, 1]
                    # Add channel encoding
                    pos_enc = torch.ones(size=(c, bs, 1), dtype=torch.float32, device=src.device)
                    pos_enc = pos_enc.cumsum(0, dtype=torch.float32)  # [C, N, 1]
                    q = k = src + pos_enc
                    src = self.self_att(q, k, value=src)[0]
                    src = self.activation(src)
                    src = src.expand_as(src1)  # [C, N, H*W]
                    excitation =src.permute(1,0,2).reshape((bs, c, h, w))
                elif self.exp_name == 'self_local_spatial_ch_seperate_att':
                    # Channel Attention
                    src1 = src.flatten(2).permute(1, 0, 2)  # [C, N, H*W]
                    src = torch.mean(src1,2).unsqueeze(2)  # [C, N, 1]
                    # Add channel encoding
                    pos_enc = torch.ones(size=(c, bs, 1), dtype=torch.float32, device=src.device)
                    pos_enc = pos_enc.cumsum(0, dtype=torch.float32)  # [C, N, 1]
                    q = k = src + pos_enc
                    src = self.self_ch_att(q, k, value=src)[0]
                    src = self.activation(src)
                    src = src.expand_as(src1)  # [C, N, H*W]
                    channel_excitation =src.permute(1,0,2).reshape((bs, c, h, w))
                    residual = residual * channel_excitation

                    # Spatial Attention
                    src = residual.detach().clone()
                    src1 = src.flatten(2).permute(2, 0, 1)  # [H*W, N, C]
                    src = torch.mean(src1,2).unsqueeze(2)  # [H*W, N, 1]
                    # Add pos encoding
                    pos_enc = torch.ones(size=(h*w, bs, 1), dtype=torch.float32, device=src.device)
                    pos_enc = pos_enc.cumsum(0, dtype=torch.float32)  # [H*W, N, 1]
                    q = k = src + pos_enc
                    src = self.self_spatial_att(q, k, value=src)[0]
                    src = self.activation(src)
                    src = src.expand_as(src1)  # [H*W, N, C]
                    excitation =src.permute(1,2,0).reshape((bs, c, h, w))

                output = (residual * excitation) + shortcut
            else:
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

        self.stage1 = self._make_stage(block, block_num[0], 64, 1, stage_num=1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2, stage_num=2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2, stage_num=3)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2, stage_num=4)

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

    def _make_stage(self, block, num, out_channels, stride, stage_num=0):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, 1, stage_num=stage_num))
        self.in_channels = out_channels * block.expansion

        for i in range(1, num):
            layers.append(block(self.in_channels, out_channels, 1, i + 1, stage_num=stage_num))

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

