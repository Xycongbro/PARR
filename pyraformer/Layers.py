import sys
import numpy as np
from torch.functional import F
from torch.functional import align_tensors
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Linear
from .SubLayers import MultiHeadAttention, PositionwiseFeedForward
import torch
from .embed import DataEmbedding, CustomEmbedding, DataEmbedding2
import math


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()
    # print(mask)
    # print(all_size)
    # # 假设mask是一个PyTorch Tensor，先转换为NumPy数组
    # mask_numpy = mask.cpu().numpy() if mask.is_cuda else mask.numpy()
    #
    # # 创建图像并指定画布大小
    # fig, ax = plt.subplots(figsize=(11, 9))  # 调整figsize以适应显示需要
    #
    # # 显示热图
    # im = ax.imshow(mask_numpy, interpolation='nearest', cmap='coolwarm')
    # # im = ax.imshow(mask_numpy, interpolation='nearest', cmap='gray')  # 改为灰度图
    # plt.colorbar(im)  # 添加颜色条
    #
    # # 获取mask的尺寸
    # n = mask_numpy.shape[0]
    #
    # # 设置坐标轴刻度位置和标签
    # ax.set_xticks(np.arange(n) + 0.5)
    # ax.set_yticks(np.arange(n) + 0.5)
    # ax.set_xticklabels(np.arange(1, n + 1), fontsize=22)  # 设置字体大小
    # ax.set_yticklabels(np.arange(1, n + 1), fontsize=22)
    #
    # # 设置标题和坐标轴标签
    # plt.title('Attention Mask Visualization', fontsize=32)
    # plt.xlabel('Node sequence number', fontsize=24)
    # plt.ylabel('Node sequence number', fontsize=24)
    #
    # # 添加网格，并设置网格线样式
    # ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
    # ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    # ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    # ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.2)
    #
    # # 调整图像边距
    # plt.subplots_adjust(left=0.3, right=0.7, top=0.7, bottom=0.3)
    #
    # # 采用tight_layout自动调整子图参数，填充整个图形区域
    # plt.tight_layout()
    #
    # plt.savefig('attention_mask_visualization.eps', format='eps')
    #
    # plt.show()
    # sys.exit(0)
    return mask, all_size


def refer_points(all_sizes, window_size, device):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index
    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][:input_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][:all_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(input_size, window_size, stride, device):
    """
    Get the index of the key that a given query needs to attend to.
    """
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size + i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        mask[input_size + i, mask[input_size + i] < input_size] = -1
        mask[input_size + i, mask[input_size + i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size + i, window_size:(window_size + stride)] = torch.arange(stride) + i * stride
        else:
            mask[input_size + i, window_size:(window_size + second_last)] = torch.arange(second_last) + i * stride

        mask[input_size + i, -1] = i // stride + third_start
        mask[input_size + i, mask[input_size + i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start + i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        mask[third_start + i, mask[third_start + i] < third_start] = -1
        mask[third_start + i, mask[third_start + i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start + i, window_size:(window_size + stride)] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start + i, window_size:(window_size + third_last)] = input_size + torch.arange(
                third_last) + i * stride

        mask[third_start + i, -1] = i // stride + fourth_start
        mask[third_start + i, mask[third_start + i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start + i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        mask[fourth_start + i, mask[fourth_start + i] < fourth_start] = -1
        mask[fourth_start + i, mask[fourth_start + i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start + i, window_size:(window_size + stride)] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start + i, window_size:(window_size + fourth_last)] = third_start + torch.arange(
                fourth_last) + i * stride

    return mask


def get_k_q(q_k_mask):
    """
    Get the index of the query that can attend to the given key.
    """
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] == i)[0]

    return k_q_mask


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, use_tvm=False,
                 q_k_mask=None, k_q_mask=None):
        super(EncoderLayer, self).__init__()
        self.use_tvm = use_tvm
        if use_tvm:
            from .PAM_TVM import PyramidalAttention
            self.slf_attn = PyramidalAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                               normalize_before=normalize_before, q_k_mask=q_k_mask, k_q_mask=k_q_mask)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                               normalize_before=normalize_before)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        if self.use_tvm:
            enc_output = self.slf_attn(enc_input)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer_Mul(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, use_tvm=False,
                 q_k_mask=None, k_q_mask=None):
        super(DecoderLayer_Mul, self).__init__()
        self.use_tvm = use_tvm
        if use_tvm:
            from .PAM_TVM import PyramidalAttention
            self.slf_attn = PyramidalAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                               normalize_before=normalize_before, q_k_mask=q_k_mask, k_q_mask=k_q_mask)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                               normalize_before=normalize_before)
            self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout,
                                            normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, cross_attn_mask=None):
        if self.use_tvm:
            dec_output = self.slf_attn(dec_input)
            dec_slf_attn = None
            dec_cross_attn = None
        else:
            dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
            dec_output, dec_cross_attn = self.cross_attn(dec_output, enc_output, enc_output, mask=cross_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_cross_attn


class DecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, Q, K, V, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            Q, K, V, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv_Construct(nn.Module):
    """Convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size)
            ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size[0]),
                ConvLayer(d_model, window_size[1]),
                ConvLayer(d_model, window_size[2])
            ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
            ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)
        all_inputs = self.norm(all_inputs)
        return all_inputs


class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(MaxPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size)
            ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size[0]),
                nn.MaxPool1d(kernel_size=window_size[1]),
                nn.MaxPool1d(kernel_size=window_size[2])
            ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(AvgPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size)
            ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.AvgPool1d(kernel_size=window_size[0]),
                nn.AvgPool1d(kernel_size=window_size[1]),
                nn.AvgPool1d(kernel_size=window_size[2])
            ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Predictor(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, num_types):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_types)

        # 初始化权重
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt, mask):
        super().__init__()
        self.model_type = opt.model
        self.mask = mask

        self.layers = nn.ModuleList([
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout,
                         normalize_before=False),
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout,
                         normalize_before=False)
        ])

        if opt.embed_type == 'CustomEmbedding':
            self.dec_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.dec_embedding = DataEmbedding2(opt.enc_in, opt.sta_in, opt.d_model, opt.dropout)

    def forward(self, x_dec, x_mark_dec, refer):
        dec_enc = self.dec_embedding(x_dec, x_mark_dec)
        dec_enc, _ = self.layers[0](dec_enc, refer, refer)
        refer_enc = torch.cat([refer, dec_enc], dim=1)
        mask = self.mask.repeat(len(dec_enc), 1, 1).to(dec_enc.device)
        dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)
        return dec_enc
