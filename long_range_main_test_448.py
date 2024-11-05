import argparse
import sys

import numpy as np
import time

import pandas as pd
import torch
import torch.optim as optim
import pyraformer.Pyraformer_LR as Pyraformer
from tqdm import tqdm
from data_loader import *
from utils.tools import TopkMSELoss, metric, MixedLoss
import matplotlib.pyplot as plt


def prepare_dataloader(args):
    """ Load data and prepare dataloader. """

    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'elect': Dataset_Custom,
        'flow': Dataset_Custom,
        'synthetic': Dataset_Synthetic,
        'camel': Dataset_Camels
    }
    Data = data_dict[args.data]

    # prepare training dataset and dataloader
    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size
    train_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.input_size, args.predict_step],
        inverse=args.inverse,
        dataset=args.data,
        sta_in=args.sta_in
    )
    print('train', len(train_set))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)

    # prepare testing dataset and dataloader
    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size
    test_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='test',
        size=[args.input_size, args.predict_step],
        inverse=args.inverse,
        dataset=args.data,
        sta_in=args.sta_in
    )
    print('test', len(test_set))
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last
    )
    return train_loader, train_set, test_loader, test_set


def sample_mining_scheduler(epoch, batch_size):
    if epoch < 2:
        topk = batch_size
    elif epoch < 4:
        topk = int(batch_size * (5 - epoch) / (6 - epoch))
    else:
        topk = int(0.5 * batch_size)

    return topk


def dataset_parameters(args, dataset):
    """Prepare specific parameters for different datasets"""
    dataset2enc_in = {
        'ETTh1': 7,
        'ETTh2': 7,
        'ETTm1': 7,
        'ETTm2': 7,
        'elect': 1,
        'flow': 1,
        'synthetic': 1,
        'camel': 6
    }
    dataset2cov_size = {
        'ETTh1': 4,
        'ETTh2': 4,
        'ETTm1': 4,
        'ETTm2': 4,
        'elect': 3,
        'flow': 3,
        'synthetic': 3,
        'camel': 3
    }
    dataset2seq_num = {
        'ETTh1': 1,
        'ETTh2': 1,
        'ETTm1': 1,
        'ETTm2': 1,
        'elect': 321,
        'flow': 1077,
        'synthetic': 60,
        'camel': 1
    }
    dataset2embed = {
        'ETTh1': 'DataEmbedding',
        'ETTh2': 'DataEmbedding',
        'ETTm1': 'DataEmbedding',
        'ETTm2': 'DataEmbedding',
        'elect': 'CustomEmbedding',
        'flow': 'CustomEmbedding',
        'synthetic': 'CustomEmbedding',
        'camel': 'DataEmbedding'
    }
    dataset2sta_in = {
        'camel': 9
    }
    args.enc_in = dataset2enc_in[dataset]
    args.sta_in = dataset2sta_in[dataset]
    args.dec_in = dataset2enc_in[dataset]
    args.covariate_size = dataset2cov_size[dataset]
    args.seq_num = dataset2seq_num[dataset]
    args.embed_type = dataset2embed[dataset]

    return args


def evaluate(model, opt):
    """ prepare dataloader """
    _, _, test_dataloader, test_dataset = prepare_dataloader(opt)

    """ Epoch operation in evaluation phase. """
    model.eval()
    preds = []
    trues = []
    nses = []
    bias = []
    apt = []
    metrics_list = [[] for _ in range(opt.predict_step)]
    with torch.no_grad():
        for batch in tqdm(test_dataloader, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            batch_x, batch_y, batch_x_mark, batch_y_mark, mean, std, location = map(lambda x: x.float().to(opt.device), batch)
            dec_inp = torch.zeros_like(batch_y).float()
            # dec_inp = torch.zeros(batch_y.shape[0], batch_y.shape[1], 1).float()

            dec_inp[:, :, :-1] = batch_y[:, :, :-1]

            # forward
            if opt.decoder == 'FC':
                # Add a predict token into the history sequence
                # predict_token = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device)
                predict_token = torch.zeros(batch_x.size(0), opt.predict_step, batch_x.size(-1), device=batch_x.device)
                predict_token[:, :, :-1] = batch_y[:, :, :-1]

                batch_x = torch.cat([batch_x, predict_token], dim=1)
                # batch_x = torch.cat([batch_x[:, :, :-1], predict_token[:, :, :-1]], dim=1)

                # batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
                batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, :, :]], dim=1)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)

            # if inverse, both the output and the ground truth are denormalized.
            if opt.inverse:
                outputs, batch_y = test_dataset.inverse_transform(outputs, batch_y, mean, std)
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    # 提取preds,trues的最后一维的最后一列
    preds_extracted = preds[:, :, -1]
    trues_extracted = trues[:, :, -1]

    # 假设trues的形状适合直接与preds_extracted比较，否则你需要对trues进行相应的提取
    # print('test shape:{}'.format(preds.shape))
    # print('test shape:{}'.format(trues_extracted.shape))
    # mae, mse, rmse, mape, mspe, nse = metric(preds, trues)

    for i in range(opt.predict_step):
        mse, mae, rmse, mape, mspe, nse, bias, atpe, kge = metric(preds_extracted[:, i], trues_extracted[:, i])
        # metrics_list[i].append((nse, rmse, atpe, bias, mse, kge))
        metrics_list[i].append((nse, rmse, atpe, bias, kge))
    return metrics_list


def parse_args():
    parser = argparse.ArgumentParser()

    # running mode
    parser.add_argument('-eval', action='store_true', default=False)

    # Path parameters
    parser.add_argument('-data', type=str, default='ETTh1')
    parser.add_argument('-root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('-data_path', type=str, default='ETTh1.csv', help='data file')

    # Dataloader parameters.
    parser.add_argument('-input_size', type=int, default=168)
    parser.add_argument('-predict_step', type=int, default=168)
    parser.add_argument('-inverse', action='store_true', help='denormalize output data', default=False)

    # Architecture selection.
    parser.add_argument('-model', type=str, default='Pyraformer')
    parser.add_argument('-decoder', type=str, default='FC')  # selection: [FC, attention]

    # Training parameters.
    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-pretrain', action='store_true', default=False)
    parser.add_argument('-hard_sample_mining', action='store_true', default=False)
    parser.add_argument('-dropout', type=float, default=0.05)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-lr_step', type=float, default=0.1)

    # Common Model parameters.
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=128)
    parser.add_argument('-d_v', type=int, default=128)
    parser.add_argument('-d_bottleneck', type=int, default=128)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layer', type=int, default=4)

    # Pyraformer parameters.
    parser.add_argument('-window_size', type=str, default='[4, 4, 4]')  # The number of children of a parent node.
    parser.add_argument('-inner_size', type=int, default=3)  # The number of ajacent nodes.
    # CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
    parser.add_argument('-CSCM', type=str, default='Bottleneck_Construct')
    parser.add_argument('-truncate', action='store_true',
                        default=False)  # Whether to remove coarse-scale nodes from the attention structure
    parser.add_argument('-use_tvm', action='store_true', default=False)  # Whether to use TVM.

    # Experiment repeat times.
    parser.add_argument('-iter_num', type=int, default=5)  # Repeat number.

    opt = parser.parse_args()
    return opt


def main(opt):
    """ Main function. """
    all_metrics_list = [[] for _ in range(opt.predict_step)]  # 创建一个包含所有指标的列表
    columns = ['location', 'nse', 'rmse', 'atpe', 'bias', 'kge']  # 假设这是你的指标名称
    values_all = pd.DataFrame(columns=columns)  # 存储所有 values

    j = 0
    for camel_one in sorted(os.listdir(opt.root_path)):
        j += 1
        print(j)
        print(camel_one)
        opt.data_path = camel_one
        print('[Info] parameters: {}'.format(opt))

        if torch.cuda.is_available():
            opt.device = torch.device("cuda", 1)
        else:
            opt.device = torch.device('cpu')

        """ prepare model """
        """ load pretrained model """
        pre_mode = eval(opt.model).Model(opt)
        model_save_dir = 'models/448/best_iter_448_1_mse_cuda_1_loss_mae.pth'
        checkpoint = torch.load(model_save_dir)["state_dict"]
        pre_mode.load_state_dict(checkpoint)

        model = pre_mode.to(opt.device)
        """ number of parameters """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))

        """ train or evaluate the model """
        metrics = evaluate(model, opt)
        for i in range(opt.predict_step):
            all_metrics_list[i].extend(metrics[i])
        mean_values = []
        median_values = []
        for metrics_per_step in all_metrics_list:
            metrics_per_step_array = np.array(metrics_per_step)
            mean_values.append(np.mean(metrics_per_step_array, axis=0))  # 计算平均值
            median_values.append(np.median(metrics_per_step_array, axis=0))  # 计算中位数
        for _, median_value in enumerate(median_values):
            print(
            f"NSE_mid: {median_value[0]}, RMSE_mid: {median_value[1]}, ATPE_mid: {median_value[2]}, Bias_mid: {median_value[3]}, KGE_mid: {median_value[4]}")
        for _, mean_value in enumerate(mean_values):
            print(
                f"NSE_mean: {mean_value[0]}, RMSE_mean: {mean_value[1]}, ATPE_mean: {mean_value[2]}, Bias_mean: {mean_value[3]}, KGE_mean: {mean_value[4]}")
        new_row = {'location': camel_one.split('.')[0], 'nse': metrics[0][0][0], 'rmse': metrics[0][0][1], 'atpe': metrics[0][0][2], 'bias': metrics[0][0][3], 'kge': metrics[0][0][4]}
        values_all = values_all.append(new_row, ignore_index=True)

    mean_values = []
    median_values = []
    for metrics_per_step in all_metrics_list:
        metrics_per_step_array = np.array(metrics_per_step)
        mean_values.append(np.mean(metrics_per_step_array, axis=0))  # 计算平均值
        median_values.append(np.median(metrics_per_step_array, axis=0))  # 计算中位数
    for _, median_value in enumerate(median_values):
        print(
            f"NSE_mid: {median_value[0]}, RMSE_mid: {median_value[1]}, ATPE_mid: {median_value[2]}, Bias_mid: {median_value[3]}, KGE_mid: {median_value[4]}")
    for _, mean_value in enumerate(mean_values):
        print(
            f"NSE_mean: {mean_value[0]}, RMSE_mean: {mean_value[1]}, ATPE_mean: {mean_value[2]}, Bias_mean: {mean_value[3]}, KGE_mean: {mean_value[4]}")

    # 将DataFrame保存为CSV文件
    values_all.to_csv('map_nse.csv', index=False)


if __name__ == '__main__':
    opt = parse_args()
    opt = dataset_parameters(opt, opt.data)
    opt.window_size = eval(opt.window_size)
    main(opt)
