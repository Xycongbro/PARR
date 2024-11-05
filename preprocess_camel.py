import os.path
import sys

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import trange


def prep_data(data, covariates, data_start, train=True):
    """Divide the training sequence into windows"""
    time_len = data.shape[0]
    input_size = window_size - stride_size
    windows_per_series = np.full((num_series), (time_len - input_size) // stride_size)
    if train: windows_per_series -= (data_start + stride_size - 1) // stride_size
    else:
        covariates = covariates[-time_len:]
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    count = 0
    for series in trange(num_series):
        cov_age = zscore(np.arange(total_time - data_start[series]))  # shape:(series_len,)
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len - data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size * i + data_start[series]
            else:
                window_start = stride_size * i
            window_end = window_start + window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input[count, 1:, 0] = data[window_start:window_end - 1, series]
            x_input[count, :, 1:1 + num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0] != 0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(), nonzero_sum) + 1
                x_input[count, :, 0] = x_input[count, :, 0] / v_input[count, 0]
                if train:
                    label[count, :] = label[count, :] / v_input[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix + 'data_' + save_name, x_input)
    np.save(prefix + 'v_' + save_name, v_input)
    np.save(prefix + 'label_' + save_name, label)


def gen_covariates(times, num_covariates):
    """Get covariates"""
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.day
        covariates[i, 3] = input_time.month
    for i in range(1, num_covariates):
        covariates[:, i] = zscore(covariates[:, i])
    return covariates[:, :num_covariates]


if __name__ == '__main__':
    global save_path
    csv_path = 'data/camels.csv'
    train_start = "1980-10-01"
    train_end = "2008-09-30"
    test_start = "2008-10-01"
    test_end = "2014-09-30"
    window_size = 192
    stride_size = 24
    num_covariates = 4
    save_name = 'camel'
    save_path = os.path.join('data', save_name)

    data_frame = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    data_frame.fillna(0, inplace=True)
    columns_to_normalize = data_frame.columns
    # data_frame[columns_to_normalize] = data_frame[columns_to_normalize].apply(zscore)
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)
    train_data = data_frame[train_start:train_end].values  # shape: [seq_length, user_num] #[10227, 674]
    test_data = data_frame[test_start:test_end].values  # [2191, 674]
    data_start = (train_data != 0).argmax(axis=0)  # find first nonzero value in each time series
    total_time = data_frame.shape[0]  # 12784
    num_series = data_frame.shape[1]  # 674
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)
