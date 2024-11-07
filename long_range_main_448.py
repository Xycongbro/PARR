import argparse
import sys

import numpy as np
import time
import torch
import torch.optim as optim

import pyraformer.Pyraformer_LR as Pyraformer
from tqdm import tqdm
from data_loader import *
from utils.tools import TopkMSELoss, metric, MixedLoss, NSELoss
import matplotlib.pyplot as plt
from utils.tools import HuberLoss

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

    # prepare testing validing dataset and dataloader
    shuffle_flag = False
    drop_last = False
    batch_size = args.batch_size
    val_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='val',
        size=[args.input_size, args.predict_step],
        inverse=args.inverse,
        dataset=args.data,
        sta_in=args.sta_in
    )
    print('val', len(val_set))
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last
    )
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
    return train_loader, train_set, test_loader, test_set, val_loader, val_set


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
        'camel': 5
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


def train_epoch(model, train_dataset, training_loader, optimizer, opt, epoch):
    """ Epoch operation in training phase. """

    model.train()
    total_loss = 0
    total_pred_number = 0

    for batch in tqdm(training_loader, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        # prepare data
        batch_x, batch_y, batch_x_mark, batch_y_mark, mean, std, location = map(lambda x: x.float().to(opt.device), batch)
        # prepare predict token
        dec_inp = torch.zeros_like(batch_y).float()
        # dec_inp = torch.zeros(batch_y.shape[0], batch_y.shape[1], 1).float()
        optimizer.zero_grad()
        # forward
        if opt.decoder == 'attention':
            if opt.pretrain and epoch < 1:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, True)
                batch_y = torch.cat([batch_x, batch_y], dim=1)
            else:
                dec_inp[:, :, :-1] = batch_y[:, :, :-1]
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
        elif opt.decoder == 'FC':
            # Add a predict token into the history sequence

            # predict_token = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device)
            predict_token = torch.zeros(batch_x.size(0), opt.predict_step, batch_x.size(-1), device=batch_x.device)
            predict_token[:, :, :-1] = batch_y[:, :, :-1]

            batch_x = torch.cat([batch_x, predict_token], dim=1)
            # batch_x = torch.cat([batch_x[:, :, :-1], predict_token[:, :, :-1]], dim=1)

            # batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
            batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, :, :]], dim=1)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
        # determine the loss function
        if opt.hard_sample_mining and not (opt.pretrain and epoch < 1):
            topk = sample_mining_scheduler(epoch, batch_x.size(0))
            criterion = TopkMSELoss(topk)
        else:
            # criterion = torch.nn.MSELoss()
            # criterion = torch.nn.MSELoss(reduction='none')
            # criterion = HuberLoss(delta=100)
            criterion = torch.nn.L1Loss()
            # criterion = NSELoss()
            # criterion = MixedLoss(0)
        # if inverse, both the output and the ground truth are denormalized.
        # if opt.inverse:
            # outputs, batch_y = train_dataset.inverse_transform(outputs, batch_y, mean, std)
        # compute loss

        losses = criterion(outputs[:, :, -1], batch_y[:, :, -1])
        # losses = criterion(outputs, batch_y)
        loss = losses.mean()
        loss.backward()

        """ update parameters """
        optimizer.step()
        total_loss += losses.sum().item()
        total_pred_number += losses.numel()
    return total_loss / total_pred_number


def eval_epoch(model, test_dataset, test_loader, opt, epoch):
    """ Epoch operation in evaluation phase. """
    model.eval()
    preds = []
    trues = []
    locations = []
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=2,
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
            # if opt.inverse:
                # outputs, batch_y = test_dataset.inverse_transform(outputs, batch_y, mean, std)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            location = location.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            locations.append(location)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        locations = np.concatenate(locations, axis=0)
        # 提取preds,trues的最后一维的最后一列
        preds_extracted = preds[:, :, -1]
        trues_extracted = trues[:, :, -1]
        # 假设trues的形状适合直接与preds_extracted比较，否则你需要对trues进行相应的提取
        # print('test shape:{}'.format(preds.shape))
        # print('test shape:{}'.format(trues_extracted.shape))
        # mae, mse, rmse, mape, mspe, nse = metric(preds, trues)

        # Flatten arrays
        flat_preds = preds_extracted.reshape(-1, preds_extracted.shape[-1])
        flat_trues = trues_extracted.reshape(-1, trues_extracted.shape[-1])
        flat_locations = locations.reshape(-1)

        # Sort based on location
        sorted_indices = np.argsort(flat_locations)
        sorted_preds = flat_preds[sorted_indices]
        sorted_trues = flat_trues[sorted_indices]
        sorted_locations = flat_locations[sorted_indices]

        unique_locations = np.unique(sorted_locations)
        location_nse = {}
        location_mse = {}
        location_mae = {}
        location_rmse = {}
        location_mape = {}
        location_mspe = {}
        location_bias = {}
        location_atpe = {}

        for loc in unique_locations:
            loc_indices = np.where(sorted_locations == loc)[0]
            loc_preds = sorted_preds[loc_indices]
            loc_trues = sorted_trues[loc_indices]
            # Calculate MSE and NSE for this location
            mse, mae, rmse, mape, mspe, nse, bias, atpe, kge = metric(loc_preds, loc_trues)
            location_nse[loc] = nse
            location_mse[loc] = mse
            location_mae[loc] = mae
            location_rmse[loc] = rmse
            location_mape[loc] = mape
            location_mspe[loc] = mspe
            location_bias[loc] = bias
            location_atpe[loc] = atpe

        mse_values = list(location_mse.values())
        nse_values = list(location_nse.values())
        mae_values = list(location_mae.values())
        rmse_values = list(location_rmse.values())
        mape_values = list(location_mape.values())
        mspe_values = list(location_mspe.values())
        bias_values = list(location_bias.values())
        atpe_values = list(location_atpe.values())

        avg_mse = np.mean(mse_values)
        median_mse = np.median(mse_values)

        avg_nse = np.mean(nse_values)
        median_nse = np.median(nse_values)

        avg_mae = np.mean(mae_values)
        median_mae = np.median(mae_values)

        avg_rmse = np.mean(rmse_values)
        median_rmse = np.median(rmse_values)

        avg_mape = np.mean(mape_values)
        median_mape = np.median(mape_values)

        avg_bias = np.mean(bias_values)
        median_bias = np.median(bias_values)

        avg_mspe = np.mean(mspe_values)
        median_mspe = np.median(mspe_values)

        avg_atpe = np.mean(atpe_values)
        median_atpe = np.median(atpe_values)

        print(f"Median MSE across locations: {median_mse}")
        print(f"Average MSE across locations: {avg_mse}")
        print(f"Median NSE across locations: {median_nse}")
        print(f"Average NSE across locations: {avg_nse}")

        # Optionally, print feature-wise results
        for i in range(flat_preds.shape[1]):
            feature_mse = {}
            feature_mae = {}
            feature_nse = {}
            for loc in unique_locations:
                loc_indices = np.where(sorted_locations == loc)[0]
                loc_preds = sorted_preds[loc_indices]
                loc_trues = sorted_trues[loc_indices]
                mse, mae, rmse, mape, mspe, nse, bias, atpe, kge = metric(loc_preds[:, i], loc_trues[:, i])
                feature_mse[f'{loc}_feature_{i}'] = mse
                feature_nse[f'{loc}_feature_{i}'] = nse

            feature_mse_values = list(feature_mse.values())
            feature_nse_values = list(feature_nse.values())
            avg_feature_mse = np.mean(feature_mse_values)
            median_feature_mse = np.median(feature_mse_values)
            avg_feature_nse = np.mean(feature_nse_values)
            median_feature_nse = np.median(feature_nse_values)

            print(f"Feature {i} - Median MSE across locations: {median_feature_mse}")
            print(f"Feature {i} - Average MSE across locations: {avg_feature_mse}")
            print(f"Feature {i} - Median NSE across locations: {median_feature_nse}")
            print(f"Feature {i} - Average NSE across locations: {avg_feature_nse}")

        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 0], trues_extracted[:, 0])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 1], trues_extracted[:, 1])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 2], trues_extracted[:, 2])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 3], trues_extracted[:, 3])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 4], trues_extracted[:, 4])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 5], trues_extracted[:, 5])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 6], trues_extracted[:, 6])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted, trues_extracted)
        # print('Epoch {}, mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, nse:{}'.format(epoch + 1, mse, mae, rmse, mape, mspe,
        #                                                                            nse))

        return avg_mse, avg_mae, avg_rmse, avg_mape, avg_mspe, avg_nse
        return mse, mae, rmse, mape, mspe, nse

def test_epoch(model, test_dataset, test_loader, opt, epoch):
    """ Epoch operation in evaluation phase. """
    model.eval()
    preds = []
    trues = []
    locations = []
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=2,
                          desc='  - (Testing) ', leave=False):
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
            location = location.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)
            locations.append(location)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        locations = np.concatenate(locations, axis=0)
        # 提取preds,trues的最后一维的最后一列
        preds_extracted = preds[:, :, -1]
        trues_extracted = trues[:, :, -1]
        # 假设trues的形状适合直接与preds_extracted比较，否则你需要对trues进行相应的提取
        # print('test shape:{}'.format(preds.shape))
        # print('test shape:{}'.format(trues_extracted.shape))
        # mae, mse, rmse, mape, mspe, nse = metric(preds, trues)

        # Flatten arrays
        flat_preds = preds_extracted.reshape(-1, preds_extracted.shape[-1])
        flat_trues = trues_extracted.reshape(-1, trues_extracted.shape[-1])
        flat_locations = locations.reshape(-1)

        # Sort based on location
        sorted_indices = np.argsort(flat_locations)
        sorted_preds = flat_preds[sorted_indices]
        sorted_trues = flat_trues[sorted_indices]
        sorted_locations = flat_locations[sorted_indices]

        unique_locations = np.unique(sorted_locations)
        location_nse = {}
        location_mse = {}
        location_mae = {}
        location_rmse = {}
        location_mape = {}
        location_mspe = {}
        location_bias = {}
        location_atpe = {}

        for loc in unique_locations:
            loc_indices = np.where(sorted_locations == loc)[0]
            loc_preds = sorted_preds[loc_indices]
            loc_trues = sorted_trues[loc_indices]
            # Calculate MSE and NSE for this location
            mse, mae, rmse, mape, mspe, nse, bias, atpe, kge = metric(loc_preds, loc_trues)
            location_nse[loc] = nse
            location_mse[loc] = mse
            location_mae[loc] = mae
            location_rmse[loc] = rmse
            location_mape[loc] = mape
            location_mspe[loc] = mspe
            location_bias[loc] = bias
            location_atpe[loc] = atpe

        mse_values = list(location_mse.values())
        nse_values = list(location_nse.values())
        mae_values = list(location_mae.values())
        rmse_values = list(location_rmse.values())
        mape_values = list(location_mape.values())
        mspe_values = list(location_mspe.values())
        bias_values = list(location_bias.values())
        atpe_values = list(location_atpe.values())

        avg_mse = np.mean(mse_values)
        median_mse = np.median(mse_values)

        avg_nse = np.mean(nse_values)
        median_nse = np.median(nse_values)

        avg_mae = np.mean(mae_values)
        median_mae = np.median(mae_values)

        avg_rmse = np.mean(rmse_values)
        median_rmse = np.median(rmse_values)

        avg_mape = np.mean(mape_values)
        median_mape = np.median(mape_values)

        avg_bias = np.mean(bias_values)
        median_bias = np.median(bias_values)

        avg_mspe = np.mean(mspe_values)
        median_mspe = np.median(mspe_values)

        avg_atpe = np.mean(atpe_values)
        median_atpe = np.median(atpe_values)

        print(f"Median MSE across locations: {median_mse}")
        print(f"Average MSE across locations: {avg_mse}")
        print(f"Median NSE across locations: {median_nse}")
        print(f"Average NSE across locations: {avg_nse}")

        # Optionally, print feature-wise results
        for i in range(flat_preds.shape[1]):
            feature_mse = {}
            feature_mae = {}
            feature_nse = {}
            for loc in unique_locations:
                loc_indices = np.where(sorted_locations == loc)[0]
                loc_preds = sorted_preds[loc_indices]
                loc_trues = sorted_trues[loc_indices]
                mse, mae, rmse, mape, mspe, nse, bias, atpe, kge = metric(loc_preds[:, i], loc_trues[:, i])
                feature_mse[f'{loc}_feature_{i}'] = mse
                feature_nse[f'{loc}_feature_{i}'] = nse

            feature_mse_values = list(feature_mse.values())
            feature_nse_values = list(feature_nse.values())
            avg_feature_mse = np.mean(feature_mse_values)
            median_feature_mse = np.median(feature_mse_values)
            avg_feature_nse = np.mean(feature_nse_values)
            median_feature_nse = np.median(feature_nse_values)

            print(f"Feature {i} - Median MSE across locations: {median_feature_mse}")
            print(f"Feature {i} - Average MSE across locations: {avg_feature_mse}")
            print(f"Feature {i} - Median NSE across locations: {median_feature_nse}")
            print(f"Feature {i} - Average NSE across locations: {avg_feature_nse}")

        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 0], trues_extracted[:, 0])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 1], trues_extracted[:, 1])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 2], trues_extracted[:, 2])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 3], trues_extracted[:, 3])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 4], trues_extracted[:, 4])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 5], trues_extracted[:, 5])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted[:, 6], trues_extracted[:, 6])
        # print(nse)
        # mse, mae, rmse, mape, mspe, nse, bias, atpe = metric(preds_extracted, trues_extracted)
        # print('Epoch {}, mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, nse:{}'.format(epoch + 1, mse, mae, rmse, mape, mspe,
        #                                                                            nse))

        return avg_mse, avg_mae, avg_rmse, avg_mape, avg_mspe, avg_nse
        return mse, mae, rmse, mape, mspe, nse



def train(model, optimizer, scheduler, opt, model_save_dir_nse, model_save_dir_mse):
    """ Start training. """

    best_mse = 100000000
    best_nse = -100000000

    no_improve_epochs = 0  # 初始化没有改善的周期数
    patience = 10  # 设置早停的耐心值，即多少个epoch没有改善就停止
    flag = 0

    """ prepare dataloader """
    # *******************************************
    training_dataloader, train_dataset, test_dataloader, test_dataset, val_dataloader, val_dataset = prepare_dataloader(opt)

    # *******************************************

    """ prepare model """
    # checkpoint = torch.load(model_save_dir)["state_dict"]

    # pre_model = eval(opt.model).Model(opt)
    #
    # pre_model.load_state_dict(checkpoint)
    # model = pre_model.to('cuda:0')

    ########
    best_metrics = []
    epoch_mse = []

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_mse = train_epoch(model, train_dataset, training_dataloader, optimizer, opt, epoch_i)
        epoch_mse.append(train_mse)
        print('  - (Training) '
              'MSE: {mse: 8.5f}'
              'elapse: {elapse:3.3f} min'
              .format(mse=train_mse, elapse=(time.time() - start) / 60))

        mse, mae, rmse, mape, mspe, nse = eval_epoch(model, val_dataset, val_dataloader, opt, epoch_i)
        # test_epoch(model, test_dataset, test_dataloader, opt, epoch_i)

        scheduler.step()

        current_metrics = [float(mse), float(mae), float(rmse), float(mape), float(mspe), float(nse)]
        if nse > best_nse:
            best_nse = nse
            best_metrics = current_metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metrics": best_metrics
                },
                model_save_dir_nse
            )
            flag += 1
        if mse < best_mse:
            best_mse = mse
            best_metrics = current_metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metrics": best_metrics
                },
                model_save_dir_mse
            )
            flag += 1
        # print(best_metrics)
        if flag > 0:
            no_improve_epochs = 0  # 重置没有改善的周期数
            # flag = 0
        else:
            no_improve_epochs += 1
        print(no_improve_epochs)
        if no_improve_epochs > patience:
            print('Stopping training')
            break
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_mse, label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return best_metrics


def evaluate(model, opt, model_save_dir_nse, model_save_dir_mse):
    """Evaluate preptrained models"""
    best_mse = 100000000

    """ prepare dataloader """
    _, _, test_dataloader, test_dataset = prepare_dataloader(opt)

    """ load pretrained model """
    checkpoint = torch.load(model_save_dir_nse)["state_dict"]
    model.load_state_dict(checkpoint)

    best_metrics = []
    mse, mae, rmse, mape, mspe, nse = eval_epoch(model, test_dataset, test_dataloader, opt, 0)

    current_metrics = [float(mse), float(mae), float(rmse), float(mape), float(mspe), float(nse)]
    if best_mse > mse:
        best_mse = mse
        best_metrics = current_metrics
    return best_metrics


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


def main(opt, iter_index):
    """ Main function. """
    print('[Info] parameters: {}'.format(opt))

    if torch.cuda.is_available():
        opt.device = torch.device("cuda", 1)
    else:
        opt.device = torch.device('cpu')

    """ prepare model """
    model = eval(opt.model).Model(opt)

    model.to(opt.device)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train or evaluate the model """
    model_save_dir_nse = 'models/448/best_iter_448_1_nse_cuda_1_wo_15_7.pth'
    model_save_dir_mse = 'models/448/best_iter_448_1_mse_cuda_1_wo_15_7.pth'
    if opt.eval:
        best_metrics = evaluate(model, opt, model_save_dir_nse, model_save_dir_mse)
    else:
        """ optimizer and scheduler """
        # optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.lr)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.lr, weight_decay=1e-5)
        # optimizer = optim.RMSprop(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=opt.lr_step)
        best_metrics = train(model, optimizer, scheduler, opt, model_save_dir_nse, model_save_dir_mse)

    print('Iteration best metrics: {}'.format(best_metrics))
    return best_metrics


if __name__ == '__main__':
    opt = parse_args()
    opt = dataset_parameters(opt, opt.data)
    opt.window_size = eval(opt.window_size)
    iter_num = opt.iter_num
    all_perf = []
    for i in range(iter_num):
        metrics = main(opt, i)
        all_perf.append(metrics)
    all_perf = np.array(all_perf)
    all_perf = all_perf.mean(0)
    print('Average Metrics: {}'.format(all_perf))
