"""
Modified based on Informer.
@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and Shanghang Zhang and Jieqi Peng and Shuai Zhang and Jianxin Li and
               Hui Xiong and Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35}, number    = {12}, pages     = {11106--11115}, publisher = {{AAAI} Press}, year      = {2021},
}
"""
import sys

import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):

        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding)
        self.batchNorm = nn.BatchNorm1d(d_model)  # 添加批标准化层
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        # self.lstm1 = nn.LSTM(c_in - 1, d_model, num_layers=1)
        # self.batchNorm = nn.BatchNorm1d(d_model)
        # self.lstm2 = nn.LSTM(c_in - 5, d_model, num_layers=1)

        # self.lstm = nn.LSTM(c_in, d_model, num_layers=1)
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1))
        x = self.batchNorm(x).transpose(1, 2)  # 执行批标准化
        return x
        # output, (h_n, c_n) = self.lstm(x)
        # output1 = output1.transpose(1, 2)
        # output1 = self.batchNorm(output1)
        # output2, (h_n, c_n) = self.lstm2(x[:, :, 5:])
        # output2 = output2.transpose(1, 2)
        # output2 = self.batchNorm(output2)
        # output = output1.transpose(1, 2) + output2.transpose(1, 2)
        return output


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class StaticEmbedding(nn.Module):
    def __init__(self, sta_in, d_model, num_node):
        super(StaticEmbedding, self).__init__()
        self.sta_in = sta_in
        self.d_model = d_model
        self.rest_blocks = nn.ModuleList()
        num_rest_blocks = 4

        for _ in range(num_rest_blocks):
            rest_block = nn.Sequential(
                nn.Conv1d(sta_in, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, sta_in, 3, padding=1),
            )
            self.rest_blocks.append(rest_block)
        spp_1_size = 1
        spp_2_size = 4
        spp_3_size = 9
        spp_4_size = 16
        self.spp_1 = nn.AdaptiveAvgPool1d(spp_1_size)
        self.spp_2 = nn.AdaptiveAvgPool1d(spp_2_size)
        self.spp_3 = nn.AdaptiveAvgPool1d(spp_3_size)
        self.spp_4 = nn.AdaptiveAvgPool1d(spp_4_size)
        self.fc = nn.Sequential(
            nn.Linear(30, num_node),
            nn.ReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(sta_in, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # 或者使用 x.permute(0, 2, 1)[batch,sta_in,seq_len]
        for rest_block in self.rest_blocks:
            x = x + rest_block(x)
        spp1 = self.spp_1(x)
        spp2 = self.spp_2(x)
        spp3 = self.spp_3(x)
        spp4 = self.spp_4(x)
        # spp5 = self.spp_5(x)
        # spp6 = self.spp_6(x)
        spp_feature = torch.cat([spp1, spp2, spp3, spp4], dim=2)
        spp_feature = self.fc(spp_feature)
        x = x + spp_feature
        return self.cnn(x).transpose(1, 2)

    # def __init__(self, sta_in, d_model):
    #     super(StaticEmbedding, self).__init__()
    #     self.sta_in = sta_in
    #     self.d_model = d_model
    #
    #     # 更复杂的嵌入层，加入非线性激活函数和正则化
    #     self.embed = nn.Sequential(
    #         nn.Linear(sta_in, d_model),
    #         # nn.ReLU()
    #         # nn.LayerNorm(d_model)
    #     )
    #
    # def forward(self, x):
    #     # 应用正则化
    #     # embedded_data = self.l2_norm(embedded_data)
    #     return self.embed(x)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = 3

        self.embed = nn.Sequential(
            nn.Linear(d_inp, d_model),
        )

    def forward(self, x):
        return self.embed(x)


"""Embedding modules. The DataEmbedding is used by the ETT dataset for long range forecasting."""


class DataEmbedding(nn.Module):
    def __init__(self, c_in, sta_in=0, d_model=None, dropout=0.1, num_node=None):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TimeFeatureEmbedding(d_model)
        self.sta_in = sta_in
        # self.value_weight = nn.Parameter(torch.randn(d_model))
        # self.position_weight = nn.Parameter(torch.randn(d_model))
        # self.temporal_weight = nn.Parameter(torch.randn(d_model))
        if sta_in != 0:
            # self.static_embedding = ResidualStaticEmbedding(sta_in=sta_in, d_model=256, dropout=0.1)
            self.static_embedding = StaticEmbedding(sta_in=sta_in, d_model=d_model, num_node=num_node)
            # self.static_embedding = nn.Linear(27, d_model)
            # self.static_weight = nn.Parameter(torch.randn(d_model))
        self.batchNorm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # value_emb = self.value_embedding(x[:, :, 8:]) * self.value_weight
        # position_emb = self.position_embedding(x) * self.position_weight
        # temporal_emb = self.temporal_embedding(x_mark) * self.temporal_weight
        #
        # if self.sta_in is not None:
        #     static_emb = self.static_embedding(x[:, :, 0:8]) * self.static_weight
        #     x = value_emb + position_emb + temporal_emb + static_emb
        # else:
        #     x = value_emb + position_emb + temporal_emb
        #
        # return self.dropout(x)
            # combined_x = torch.cat((x_mark, x[:, :, 27:]), dim=2)

        # x = self.value_embedding(x[:, :, self.sta_in:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:self.sta_in])

        x = self.value_embedding(x[:, :, self.sta_in:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:self.sta_in])


        # x = self.value_embedding(x[:, :, 27:]) + self.position_embedding(x)
        # + self.temporal_embedding(x_mark))

            # x = (self.value_weight * self.value_embedding(x[:, :, 27:]) + self.position_embedding(x) * self.position_weight
            #      + self.static_embedding(x[:, :, 0:27]) * self.static_weight)

            # x = self.value_embedding(x[:, :, 8:]) + self.position_embedding(x) + self.temporal_embedding(x_mark)

            # x = (self.value_embedding(x[:, :, 8:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:8]))


        # elif self.sta_in == 21:
        #     # combined_x = torch.cat((x_mark, x[:, :, 27:]), dim=2)
        #     x = self.value_embedding(x[:, :, 21:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:21])
        #     # + self.temporal_embedding(x_mark))
        #
        #     # x = (self.value_weight * self.value_embedding(x[:, :, 27:]) + self.position_embedding(x) * self.position_weight
        #     #      + self.static_embedding(x[:, :, 0:27]) * self.static_weight)
        #
        #     # x = self.value_embedding(x[:, :, 8:]) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        #
        #     # x = (self.value_embedding(x[:, :, 8:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:8]))


        # elif self.sta_in == 7:
            # combined_x = torch.cat((x_mark, x[:, :, 27:]), dim=2)
            # x = self.value_embedding(x[:, :, 7:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:7])

        #
        # elif self.sta_in == 22:
        #     # combined_x = torch.cat((x_mark, x[:, :, 27:]), dim=2)
        #     x = self.value_embedding(x[:, :, 22:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:22])
        # else:
        #     x = self.value_embedding(x) + self.position_embedding(x)
        x = self.batchNorm(x.transpose(1, 2)).transpose(1, 2)
        # return x
        return self.dropout(x)


"""The CustomEmbedding is used by the electricity dataset and app flow dataset for long range forecasting."""


class DataEmbedding2(nn.Module):
    def __init__(self, c_in, sta_in=0, d_model=None, dropout=0.1, num_node = None):
        super(DataEmbedding2, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in - 5, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model)
        self.sta_in = sta_in
        # self.value_weight = nn.Parameter(torch.randn(d_model))
        # self.position_weight = nn.Parameter(torch.randn(d_model))
        # self.temporal_weight = nn.Parameter(torch.randn(d_model))
        # if sta_in != 0:
        #     # self.static_embedding = ResidualStaticEmbedding(sta_in=sta_in, d_model=256, dropout=0.1)
        #     self.static_embedding = StaticEmbedding(sta_in=sta_in, d_model=d_model, num_node = num_node)
        #     self.static_weight = nn.Parameter(torch.randn(d_model))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # value_emb = self.value_embedding(x[:, :, 8:]) * self.value_weight
        # position_emb = self.position_embedding(x) * self.position_weight
        # temporal_emb = self.temporal_embedding(x_mark) * self.temporal_weight
        #
        # if self.sta_in is not None:
        #     static_emb = self.static_embedding(x[:, :, 0:8]) * self.static_weight
        #     x = value_emb + position_emb + temporal_emb + static_emb
        # else:
        #     x = value_emb + position_emb + temporal_emb
        #
        # return self.dropout(x)
        if self.sta_in != 0:
            # combined_x = torch.cat((x_mark, x[:, :, 27:]), dim=2)
            x = self.value_embedding(x[:, :, 32:33]) + self.position_embedding(x)
            # + self.temporal_embedding(x_mark))

            # x = (self.value_weight * self.value_embedding(x[:, :, 27:]) + self.position_embedding(x) * self.position_weight
            #      + self.static_embedding(x[:, :, 0:27]) * self.static_weight)

            # x = self.value_embedding(x[:, :, 8:]) + self.position_embedding(x) + self.temporal_embedding(x_mark)

            # x = (self.value_embedding(x[:, :, 8:]) + self.position_embedding(x) + self.static_embedding(x[:, :, 0:8]))
        else:
            x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        super(CustomEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark[:, :, :-1]) \
            + self.seqid_embedding(x_mark[:, :, -1].long())

        return self.dropout(x)


"""The SingleStepEmbedding is used by all datasets for single step forecasting."""


class SingleStepEmbedding(nn.Module):
    def __init__(self, cov_size, num_seq, d_model, input_size, device):
        super().__init__()

        self.cov_size = cov_size
        self.num_class = num_seq
        self.cov_emb = nn.Linear(cov_size + 1, d_model)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.data_emb = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=padding,
                                  padding_mode='circular')

        self.position = torch.arange(input_size, device=device).unsqueeze(0)
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
                                         device=device)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transformer_embedding(self, position, vector):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = position.unsqueeze(-1) / vector
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x):
        covs = x[:, :, 1:(1 + self.cov_size)]
        seq_ids = ((x[:, :, -1] / self.num_class) - 0.5).unsqueeze(2)
        covs = torch.cat([covs, seq_ids], dim=-1)
        cov_embedding = self.cov_emb(covs)
        data_embedding = self.data_emb(x[:, :, 0].unsqueeze(2).permute(0, 2, 1)).transpose(1, 2)
        embedding = cov_embedding + data_embedding

        position = self.position.repeat(len(x), 1).to(x.device)
        position_emb = self.transformer_embedding(position, self.position_vec.to(x.device))

        embedding += position_emb

        return embedding
