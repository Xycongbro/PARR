import csv
import pickle
import sys

import pandas as pd
from torch import nn
from torch.nn.modules import loss
import torch
import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def NSE(pred, true):
    observation_bar = np.mean(true)  # 计算真实值的平均值
    # print(np.sum(pred < 0))
    # print(pred.shape)
    numerator = np.sum((true - pred) ** 2)  # 计算分子，即预测值与真实值差值的平方和
    denominator = np.sum((true - observation_bar) ** 2)  # 计算分母，即真实值与其平均值差值的平方和
    # numerator = np.sum((np.log(true/pred)) ** 2)  # 计算分子，即预测值与真实值差值的平方和
    # denominator = np.sum((np.log(true/observation_bar)) ** 2)  # 计算分母，即真实值与其平均值差值的平方和
    # 计算NSE
    nse = 1 - numerator / denominator
    if nse < 0:
        nse = 0
    return nse
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = torch.abs(error) <= self.delta
        small_error_loss = 0.5 * error**2
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.where(is_small_error, small_error_loss, large_error_loss).mean()

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

def ATPE(pred, true):
    n = len(true)
    top_2_percent_index = int(n * 0.02)
    # 获取真实值中前2%的值及其索引
    sorted_indexes = np.argsort(true)
    top_2_percent_indexes = sorted_indexes[-top_2_percent_index:]
    # 提取对应的预测值
    top_2_percent_predicted_values = pred[top_2_percent_indexes]
    # 计算前2%真实值的总和
    sum_of_true_values = np.sum(true[top_2_percent_indexes])
    atpe = np.sum(np.abs(true[top_2_percent_indexes] - top_2_percent_predicted_values)) / sum_of_true_values
    return atpe


def BIAS(pred, true):
    return np.sum(pred - true) / np.sum(true)

def KGE(pred, true):
    pred_mean = np.mean(pred)
    true_mean = np.mean(true)
    pred_std = np.std(pred)
    true_std = np.std(true)
    beta = pred_mean / true_mean
    alpha = pred_std / true_std
    numerator = np.mean(((true - true_mean) * (pred - pred_mean)))
    denominator = true_std * pred_std
    gamma = numerator / denominator
    kge = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (gamma - 1) ** 2)
    if kge < 0:
        kge = 0
    return kge

def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    nse = NSE(pred, true)
    atpe = ATPE(pred, true)
    bias = BIAS(pred, true)
    kge = KGE(pred, true)
    return mse, mae, rmse, mape, mspe, nse, bias, atpe, kge


class NSELoss(nn.Module):
    def __init__(self, eps: float = 0.1, delta=1.0):
        super().__init__()
        eps = torch.tensor(eps, dtype=torch.float32)
        self.eps = eps
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true) ** 2
        self.eps = self.eps.to(q_stds.device)
        weights = 1 / (q_stds + self.eps) ** 2
        weights = weights.reshape(-1, 1, 1)
        scaled_loss = weights * squared_error
        # mae = torch.abs(y_true - y_pred)
        # mask = mae < self.delta
        # loss = torch.where(mask, scaled_loss, mae)
        # return torch.mean(loss)
        return torch.mean(scaled_loss)


class MixedLoss(nn.Module):
    def __init__(self, alpha):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='none')
        self.nse_loss = NSELoss()

    def forward(self, y_true, y_pred, std=None):
        mse_losses = self.mse_loss(y_true, y_pred)
        nse_losses = self.nse_loss(y_true, y_pred, std)
        mixed_losses = self.alpha * mse_losses + (1 - self.alpha) * abs(nse_losses - 1)
        loss = mixed_losses.mean()
        return loss

class SmoothNSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(SmoothNSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        # 计算观测值的均值
        mean_observed = torch.mean(targets, dim=0)

        # 计算分子部分 (sum of squared errors)
        numerator = torch.sum((predictions - targets) ** 2, dim=0)

        # 计算分母部分 (sum of squared deviations from mean)
        denominator = torch.sum((targets - mean_observed) ** 2, dim=0) + self.epsilon

        # 计算 NSE 并进行平滑
        nse = 1 - (numerator / denominator)

        # 取负值作为损失，因为我们希望最大化 NSE
        loss = -torch.mean(nse)
        return loss


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def save(self, filename):
        # 保存均值和标准差到CSV文件
        data = {
            'mean': self.mean.tolist() if torch.is_tensor(self.mean) else self.mean,
            'std': self.std.tolist() if torch.is_tensor(self.std) else self.std
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        # df = pd.DataFrame(data)
        # df.to_csv(filename, index=False)
        print(f"均值和标准差已保存到 {filename}")

    def load(self, filename):
        # 从CSV文件加载均值和标准差
        # df = pd.read_csv(filename)
        # 从pkl文件加载均值和标准差
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # self.mean = torch.tensor(df['mean'].values) if torch.is_tensor(self.mean) else df['mean'].values
        # self.std = torch.tensor(df['std'].values) if torch.is_tensor(self.std) else df['std'].values
        self.mean = torch.tensor(data['mean']) if isinstance(data['mean'], list) else data['mean']
        self.std = torch.tensor(data['std']) if isinstance(data['std'], list) else data['std']
        print(f"均值和标准差已从 {filename} 加载")

    def transform(self, data, number=0, isStatic=True):
        if number == 0:
            data = torch.from_numpy(data)  # 将输入数据转换为张量
            mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
            std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
            normalized_data = (data - mean) / std
            return normalized_data
        # 448
        elif isStatic:
            data = torch.from_numpy(data)  # 将输入数据转换为张量
            mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
            std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
            std[std == 0] = 1e-15  # 将标准差为0的元素替换为一个很小的值，比如1e-15
            cols_to_exclude = data[:, :number]
            cols_to_normalize = data[:, number:]
            normalized_data = (cols_to_normalize - mean[number:]) / std[number:]
            transformed_data = torch.cat((cols_to_exclude, normalized_data), dim=1)
            return transformed_data
        # elif isStatic and number == 296:
        #     data = torch.from_numpy(data)  # 将输入数据转换为张量
        #     mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        #     std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        #     std[std == 0] = 1e-15  # 将标准差为0的元素替换为一个很小的值，比如1e-15
        #     cols_to_exclude = data[:, :21]
        #     cols_to_normalize = data[:, 21:]
        #     normalized_data = (cols_to_normalize - mean[21:]) / std[21:]
        #     transformed_data = torch.cat((cols_to_exclude, normalized_data), dim=1)
        #     return transformed_data
        # elif isStatic and number == 897:
        #     data = torch.from_numpy(data)  # 将输入数据转换为张量
        #     mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        #     std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        #     std[std == 0] = 1e-15  # 将标准差为0的元素替换为一个很小的值，比如1e-15
        #     cols_to_exclude = data[:, :7]
        #     cols_to_normalize = data[:, 7:]
        #     normalized_data = (cols_to_normalize - mean[7:]) / std[7:]
        #     transformed_data = torch.cat((cols_to_exclude, normalized_data), dim=1)
        #     return transformed_data
        # elif isStatic and number == "gb":
        #     data = torch.from_numpy(data)  # 将输入数据转换为张量
        #     mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        #     std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        #     std[std == 0] = 1e-15  # 将标准差为0的元素替换为一个很小的值，比如1e-15
        #     cols_to_exclude = data[:, :22]
        #     cols_to_normalize = data[:, 22:]
        #     normalized_data = (cols_to_normalize - mean[22:]) / std[22:]
        #     transformed_data = torch.cat((cols_to_exclude, normalized_data), dim=1)
        #     return transformed_data
        # # 673
        # else:
        #     data = torch.from_numpy(data)  # 将输入数据转换为张量
        #     mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        #     std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        #     std[std == 0] = 1e-15  # 将标准差为0的元素替换为一个很小的值，比如1e-15
        #     return (data - mean) / std

    def inverse_transform(self, data, seq_y, mean, std):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std[-1]) + mean[-1], (seq_y * std) + mean


class TopkMSELoss(torch.nn.Module):
    def __init__(self, topk) -> None:
        super().__init__()
        self.topk = topk
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, output, label):
        losses = self.criterion(output, label).mean(2).mean(1)
        losses = torch.topk(losses, self.topk)[0]

        return losses


class SingleStepLoss(torch.nn.Module):
    """ Compute top-k log-likelihood and mse. """

    def __init__(self, ignore_zero):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(self, mu, sigma, labels, topk=0):
        if self.ignore_zero:
            indexes = (labels != 0)
        else:
            indexes = (labels >= 0)

        distribution = torch.distributions.normal.Normal(mu[indexes], sigma[indexes])
        likelihood = -distribution.log_prob(labels[indexes])

        diff = labels[indexes] - mu[indexes]
        se = diff * diff

        if 0 < topk < len(likelihood):
            likelihood = torch.topk(likelihood, topk)[0]
            se = torch.topk(se, topk)[0]

        return likelihood, se


def AE_loss(mu, labels, ignore_zero):
    if ignore_zero:
        indexes = (labels != 0)
    else:
        indexes = (labels >= 0)

    ae = torch.abs(labels[indexes] - mu[indexes])
    return ae

