import sys

import torch
import torch.nn as nn


class FeatureProcessingModule(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super(FeatureProcessingModule, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, padding_mode='circular'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),  # 添加 Dropout 层
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 将输入特征的维度顺序调整为 [batch_size, channels, sequence_length]
        x = x.transpose(1, 2)
        # 应用 1D 卷积操作
        # 将输出特征的维度顺序调整为 [batch_size, sequence_length, channels]
        x = self.embed(x).transpose(1, 2)
        return x
