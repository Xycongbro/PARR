import os
import pickle
import sys
import time
import pandas as pd
from scipy.stats import zscore

from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler
from utils.timefeatures import time_features
import numpy as np
import torch
import bisect
import warnings

warnings.filterwarnings('ignore')

"""Long range dataloader"""


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', dataset='ETTh1', inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=1, freq='h')

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, seq_y, mean, std):
        return self.scaler.inverse_transform(data), seq_y


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTm1.csv', dataset='ETTm1', inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=1, freq='h')

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, seq_y, mean, std):
        return self.scaler.inverse_transform(data), seq_y


"""Long range dataloader for camel dataset"""


class Dataset_Camel(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTm1.csv', dataset='camel', inverse=False):
        # size [input_size, predict_step]

        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        self.date_column = 'date'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        train_start_date = '1980-01-01'
        val_start_date = '1999-01-01'
        test_start_date = '2004-01-01'
        test_end_date = '2008-12-31'
        df_raw[self.date_column] = pd.to_datetime(df_raw[self.date_column], errors='coerce')
        train_start_idx = df_raw.index[df_raw[self.date_column] == train_start_date][0]
        val_start_idx = df_raw.index[df_raw[self.date_column] == val_start_date][0]
        test_start_idx = df_raw.index[df_raw[self.date_column] == test_start_date][0]
        test_end_idx = df_raw.index[df_raw[self.date_column] == test_end_date][0]

        border1s = [train_start_idx, val_start_idx - self.seq_len, test_start_idx - self.seq_len]
        border2s = [val_start_idx, test_start_idx, test_end_idx]
        # border1s = [0, 6940 - self.seq_len, 8766 - self.seq_len]
        # border2s = [6940, 8766, 10592]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[2:]
        df_data = df_raw[cols_data]
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=0, freq='d')
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data, seq_y, mean, std):
        return self.scaler.inverse_transform(data), seq_y

class Dataset_Camels(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTm1.csv', dataset='camel', inverse=False,
                 locations=None, location_column="location", date_column='date', sta_in=None):
        # size [input_size, predict_step]
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'val', 'test']
        if data_path == 'camels_448.csv' or data_path == 'camels_448_train.csv' or root_path == 'data/448':
            type_map = {'train': 2, 'val': 1, 'test': 0}
        else:
            type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.sta_in = sta_in
        self.inverse = inverse
        self.date_column = date_column
        self.location_column = location_column
        self.locations = locations

        self.root_path = root_path
        self.data_path = data_path
        self.data_per_location = {}  # 存储每个地点的数据

        self.scaler = StandardScaler()
        self.__read_data__(flag)
        self.cumulative_lengths = self._calculate_cumulative_lengths()

    def __read_data__(self, flag, train=False):
        if train == True:
            if os.path.exists('data/locations.pkl'):
                with open('data/locations.pkl', 'rb') as f:
                    self.locations = pickle.load(f)
            else:
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
                self.locations = df_raw[self.location_column].unique()
                with open('data/locations.pkl', 'wb') as f:
                    pickle.dump(self.locations, f)
            if self.data_path == 'camels_448.csv' or self.data_path == 'camels_448_train.csv' or self.root_path == 'data/448':
                file_path = "data/normal_us_sta_9.pkl"
            elif self.data_path == 'camels_gb_train.csv' or self.data_path == 'camels_gb.csv' or self.root_path == 'data/gb':
                file_path = "data/normal_gb.pkl"
            if os.path.exists(file_path):
                self.scaler.load(file_path)
            else:
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
                train_data_all = []
                for location in self.locations:
                    df_location = df_raw[df_raw[self.location_column] == location].copy()
                    # self.process_location_data(df_location, location)
                    train_data_all.append(self.extract_train_data(df_location))
                train_data_all = pd.concat(train_data_all)
                train_data_all = train_data_all.dropna()
                self.scaler.fit(train_data_all.values)
                self.scaler.save(file_path)
            if os.path.exists('data/data_sta_9_per_location_' + flag + '.pkl'):
                self.data_per_location = pickle.load(open('data/data_sta_9_per_location_' + flag + '.pkl', 'rb'))
            else:
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
                for location in self.locations:
                    df_location = df_raw[df_raw[self.location_column] == location].copy()
                    self.process_location_data_2(df_location, location)
                    self.locations = df_raw[self.location_column].unique()
                with open('data/data_sta_9_per_location_' + flag + '.pkl', 'wb') as f:
                    pickle.dump(self.data_per_location, f)
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            self.locations = df_raw[self.location_column].unique()
            if self.data_path == 'camels_448.csv' or self.data_path == 'camels_448_train.csv' or self.root_path == 'data/448':
                file_path = "data/normal_us_sta_9.pkl"
            elif self.data_path == 'camels_gb_train.csv' or self.data_path == 'camels_gb.csv' or self.root_path == 'data/gb':
                file_path = "data/normal_gb.pkl"
            if os.path.exists(file_path):
                self.scaler.load(file_path)
            else:
                df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
                train_data_all = []
                for location in self.locations:
                    df_location = df_raw[df_raw[self.location_column] == location].copy()
                    # self.process_location_data(df_location, location)
                    train_data_all.append(self.extract_train_data(df_location))
                train_data_all = pd.concat(train_data_all)
                train_data_all = train_data_all.dropna()
                self.scaler.fit(train_data_all.values)
                self.scaler.save(file_path)
            for location in self.locations:
                df_location = df_raw[df_raw[self.location_column] == location].copy()
                self.process_location_data_2(df_location, location)
                self.locations = df_raw[self.location_column].unique()
    def extract_train_data(self, df_location):
        # if self.data_path == 'camels_448.csv' or self.data_path == 'camels_448_train.csv' or self.root_path == 'data/448':
        #     train_start_date = '2001-10-01'
        #     train_end_date = '2008-09-30'
        # elif self.data_path == 'camels_296_train.csv' or self.data_path == 'camels_296.csv' or self.root_path == 'data/296_2':
        #     train_start_date = '1981-01-01'
        #     train_end_date = '2004-12-31'
        # elif self.data_path == 'camels_897_train.csv' or self.data_path == 'camels_897.csv' or self.root_path == 'data/897':
        #     train_start_date = '1980-01-01'
        #     train_end_date = '2003-12-31'
        # elif self.data_path == 'camels_gb_train.csv' or self.data_path == 'camels_gb.csv' or self.root_path == 'data/gb':
        #     train_start_date = '1993-10-01'
        #     train_end_date = '2003-09-30'
        # else:
        #     train_start_date = '1980-10-01'
        #     train_end_date = '1994-09-30'
        #
        # train_data = df_location[(df_location[self.date_column] >= train_start_date) &
        #                          (df_location[self.date_column] <= train_end_date)]
        # cols_data = train_data.columns[2:]  # 除去日期和地点的列名
        # return train_data[cols_data]
        if self.data_path == 'camels_448.csv' or self.data_path == 'camels_448_train.csv' or self.root_path == 'data/448':
            test_start_date = '1989-10-01'
            val_start_date = '1999-10-01'
            train_start_date = '2001-10-01'
            train_end_date = '2008-09-30'
            # RR-Former
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            train_end_idx = df_location.index[df_location[self.date_column] == train_end_date][0]
            # border1s = [test_start_idx - test_start_idx, val_start_idx - self.seq_len - test_start_idx,
            #             train_start_idx - self.seq_len - test_start_idx]
            # border2s = [val_start_idx - test_start_idx, train_start_idx - test_start_idx, train_end_idx - test_start_idx]
            total_len = len(df_location)
            border1s = [test_start_idx % total_len, val_start_idx % total_len, train_start_idx % total_len]
            border2s = [val_start_idx % total_len, train_start_idx % total_len, train_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[2]:border2s[2]]
        elif self.data_path == 'camels_296_train.csv' or self.data_path == 'camels_296.csv' or self.root_path == 'data/296_2':
            train_start_date = '1981-01-01'
            val_start_date = '2005-01-01'
            test_start_date = '2013-01-01'
            test_end_date = '2020-12-31'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        elif self.data_path == 'camels_897_train.csv' or self.data_path == 'camels_897.csv' or self.root_path == 'data/897':
            train_start_date = '1980-01-01'
            val_start_date = '2004-01-01'
            test_start_date = '2011-01-01'
            test_end_date = '2018-12-31'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        elif self.data_path == 'camels_gb_train.csv' or self.data_path == 'camels_gb.csv' or self.root_path == 'data/gb':
            train_start_date = '1993-10-01'
            val_start_date = '2003-10-01'
            test_start_date = '2005-10-01'
            test_end_date = '2015-09-30'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        else:
            train_start_date = '1980-10-01'
            val_start_date = '1995-10-01'
            test_start_date = '2000-10-01'
            test_end_date = '2014-09-30'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            # df_location.fillna(0, inplace=True)

            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        return train_data
    def process_location_data_2(self, df_location, location):
        # 根据数据集类型(train/val/test)分割数据
        # 448
        # RR-Former
        if self.data_path == 'camels_448.csv' or self.data_path == 'camels_448_train.csv' or self.root_path == 'data/448':
            test_start_date = '1989-10-01'
            val_start_date = '1999-10-01'
            train_start_date = '2001-10-01'
            train_end_date = '2008-09-30'
            # RR-Former
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            train_end_idx = df_location.index[df_location[self.date_column] == train_end_date][0]
            # border1s = [test_start_idx - test_start_idx, val_start_idx - self.seq_len - test_start_idx,
            #             train_start_idx - self.seq_len - test_start_idx]
            # border2s = [val_start_idx - test_start_idx, train_start_idx - test_start_idx, train_end_idx - test_start_idx]
            total_len = len(df_location)
            border1s = [test_start_idx % total_len, val_start_idx % total_len, train_start_idx % total_len]
            border2s = [val_start_idx % total_len, train_start_idx % total_len, train_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[2]:border2s[2]]
        elif self.data_path == 'camels_296_train.csv' or self.data_path == 'camels_296.csv' or self.root_path == 'data/296_2':
            train_start_date = '1981-01-01'
            val_start_date = '2005-01-01'
            test_start_date = '2013-01-01'
            test_end_date = '2020-12-31'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        elif self.data_path == 'camels_897_train.csv' or self.data_path == 'camels_897.csv' or self.root_path == 'data/897':
            train_start_date = '1980-01-01'
            val_start_date = '2004-01-01'
            test_start_date = '2011-01-01'
            test_end_date = '2018-12-31'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        elif self.data_path == 'camels_gb_train.csv' or self.data_path == 'camels_gb.csv' or self.root_path == 'data/gb':
            train_start_date = '1993-10-01'
            val_start_date = '2003-10-01'
            test_start_date = '2005-10-01'
            test_end_date = '2015-09-30'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        else:
            train_start_date = '1980-10-01'
            val_start_date = '1995-10-01'
            test_start_date = '2000-10-01'
            test_end_date = '2014-09-30'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            # df_location.fillna(0, inplace=True)

            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        data_x_init = df_data[border1:border2]
        data_stamp = df_location[[self.date_column]][border1:border2]
        data_stamp[self.date_column] = pd.to_datetime(data_stamp.date)
        # 删除非法数据
        # 过滤掉所有包含0的行
        # data_x_init = data_x_init[data_x_init.iloc[:, -1] != 0]
        data_x_init = data_x_init.dropna()
        data_stamp = data_stamp[data_stamp.index.isin(data_x_init.index)]
        data_stamp = time_features(data_stamp, timeenc=0, freq='d')
        # 整合数据
        data_x = self.scaler.transform(data_x_init.values, self.sta_in, True)[:]
        # data_y = data_x_init.values[border1:border2] if self.inverse else data_x

        # data_y = data_x_init.values if self.inverse else data_x
        data_y = data_x
        self.data_per_location[location] = {
            'data_x': data_x,
            'data_y': data_y,
            'data_stamp': data_stamp
        }

    def process_location_data(self, df_location, location):
        # 根据数据集类型(train/val/test)分割数据
        # 448
        # RR-Former
        if self.data_path == 'camels_448.csv' or self.data_path == 'camels_448_train.csv' or self.root_path == 'data/448':
            test_start_date = '1989-10-01'
            val_start_date = '1999-10-01'
            train_start_date = '2001-10-01'
            train_end_date = '2008-09-30'
            # RR-Former
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            train_end_idx = df_location.index[df_location[self.date_column] == train_end_date][0]
            # border1s = [test_start_idx - test_start_idx, val_start_idx - self.seq_len - test_start_idx,
            #             train_start_idx - self.seq_len - test_start_idx]
            # border2s = [val_start_idx - test_start_idx, train_start_idx - test_start_idx, train_end_idx - test_start_idx]
            total_len = len(df_location)
            border1s = [test_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (train_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, train_start_idx % total_len, train_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[2]:border2s[2]]
        elif self.data_path == 'camels_296_train.csv' or self.data_path == 'camels_296.csv' or self.root_path == 'data/296_2':
            train_start_date = '1981-01-01'
            val_start_date = '2005-01-01'
            test_start_date = '2013-01-01'
            test_end_date = '2020-12-31'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        elif self.data_path == 'camels_897_train.csv' or self.data_path == 'camels_897.csv' or self.root_path == 'data/897':
            train_start_date = '1980-01-01'
            val_start_date = '2004-01-01'
            test_start_date = '2011-01-01'
            test_end_date = '2018-12-31'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        elif self.data_path == 'camels_gb_train.csv' or self.data_path == 'camels_gb.csv' or self.root_path == 'data/gb':
            train_start_date = '1993-10-01'
            val_start_date = '2003-10-01'
            test_start_date = '2005-10-01'
            test_end_date = '2015-09-30'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            # df_location.fillna(0, inplace=True)
            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]
        else:
            train_start_date = '1980-10-01'
            val_start_date = '1995-10-01'
            test_start_date = '2000-10-01'
            test_end_date = '2014-09-30'
            # RR-Former
            train_start_idx = df_location.index[df_location[self.date_column] == train_start_date][0]
            val_start_idx = df_location.index[df_location[self.date_column] == val_start_date][0]
            test_start_idx = df_location.index[df_location[self.date_column] == test_start_date][0]
            test_end_idx = df_location.index[df_location[self.date_column] == test_end_date][0]
            total_len = len(df_location)
            border1s = [train_start_idx % total_len, (val_start_idx - self.seq_len) % total_len,
                        (test_start_idx - self.seq_len) % total_len]
            border2s = [val_start_idx % total_len, test_start_idx % total_len, test_end_idx % total_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            # df_location.fillna(0, inplace=True)

            cols_data = df_location.columns[2:]  # 除去日期和地点的列名
            df_data = df_location[cols_data]  # 其余列的原数据
            train_data = df_data[border1s[0]:border2s[0]]

        ##################
        # 标准化
        # train_data = train_data[train_data.iloc[:, -1] != 0]
        train_data = train_data.dropna()
        self.scaler.fit(train_data.values)
        # 提取原始数据
        data_x_init = df_data[border1:border2]

        data_stamp = df_location[[self.date_column]][border1:border2]
        data_stamp[self.date_column] = pd.to_datetime(data_stamp.date)
        # 删除非法数据
        # 过滤掉所有包含0的行
        # data_x_init = data_x_init[data_x_init.iloc[:, -1] != 0]
        data_x_init = data_x_init.dropna()
        data_stamp = data_stamp[data_stamp.index.isin(data_x_init.index)]
        data_stamp = time_features(data_stamp, timeenc=0, freq='d')
        # 整合数据
        if self.data_path == 'camels_448.csv' or self.data_path == 'camels_448_train.csv' or self.root_path == 'data/448':
            data_x = self.scaler.transform(data_x_init.values, 448, True)[:]
        elif self.data_path == 'camels_296.csv' or self.data_path == 'camels_296_train.csv' or self.root_path == 'data/296_2':
            data_x = self.scaler.transform(data_x_init.values, 296, True)[:]
        elif self.data_path == 'camels_897.csv' or self.data_path == 'camels_897_train.csv' or self.root_path == 'data/897':
            data_x = self.scaler.transform(data_x_init.values, 897, True)[:]
        elif self.data_path == 'camels_gb.csv' or self.data_path == 'camels_gb_train.csv' or self.root_path == 'data/gb':
            data_x = self.scaler.transform(data_x_init.values, "gb", True)[:]
        else:
            data_x = self.scaler.transform(data_x_init.values, 673, False)[:]
        # data_y = data_x_init.values[border1:border2] if self.inverse else data_x
        data_y = data_x_init.values if self.inverse else data_x
        self.data_per_location[location] = {
            'data_x': data_x,
            'data_y': data_y,
            'data_stamp': data_stamp
        }
    def _calculate_cumulative_lengths(self):
        cumulative_lengths = [0]
        for location in self.locations:
            location_data = self.data_per_location[location]
            data_length = len(location_data['data_x'])
            cumulative_lengths.append(cumulative_lengths[-1] + data_length - self.seq_len - self.pred_len + 1)
        return cumulative_lengths

    def __getitem__(self, index):
        total_length = self.cumulative_lengths[-1]
        if index >= total_length or index < 0:
            raise IndexError("Index out of range")

        location_index = bisect.bisect_right(self.cumulative_lengths, index) - 1
        location = self.locations[location_index]
        location_data = self.data_per_location[location]
        data_length = len(location_data['data_x'])
        max_len = data_length - self.seq_len - self.pred_len
        s_begin = index - self.cumulative_lengths[location_index]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = location_data['data_x'][s_begin:s_end]
        seq_y = location_data['data_y'][r_begin:r_end]
        seq_x_mark = location_data['data_stamp'][s_begin:s_end]
        seq_y_mark = location_data['data_stamp'][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std, location

        # total_length = sum([len(self.data_per_location[loc]['data_x']) for loc in self.locations])
        # if index >= total_length or index < 0:
        #     raise IndexError("Index out of range")
        # for location in self.locations:
        #     location_data = self.data_per_location[location]
        #     data_length = len(location_data['data_x'])
        #     max_len = data_length - self.seq_len - self.pred_len
        #     if index <= max_len:
        #         s_begin = index
        #         s_end = s_begin + self.seq_len
        #         r_begin = s_end
        #         r_end = r_begin + self.pred_len
        #         seq_x = location_data['data_x'][s_begin:s_end]
        #         seq_y = location_data['data_y'][r_begin:r_end]
        #         seq_x_mark = location_data['data_stamp'][s_begin:s_end]
        #         seq_y_mark = location_data['data_stamp'][r_begin:r_end]
        #         return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std
        #     index -= max_len + 1
        # raise IndexError("Index out of range")


        # location = self.locations[index % len(self.locations)]
        # location_data = self.data_per_location[location]
        # s_begin = index // len(self.locations)
        # s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len
        #
        # seq_x = location_data['data_x'][s_begin:s_end]
        # seq_y = location_data['data_y'][r_begin:r_end]
        # seq_x_mark = location_data['data_stamp'][s_begin:s_end]
        # seq_y_mark = location_data['data_stamp'][r_begin:r_end]
        # return seq_x, seq_y, seq_x_mark, seq_y_mark, self.scaler.mean, self.scaler.std

    def __len__(self):
        return self.cumulative_lengths[-1]

        # total_length = 0
        # for location in self.locations:
        #     # 计算每个地点的数据长度
        #     data_length = len(self.data_per_location[location]['data_x'])
        #     # 对于每个地点，计算可以提取的序列数量
        #     if data_length >= self.seq_len + self.pred_len:
        #         total_length += data_length - self.seq_len - self.pred_len + 1
        # return total_length

    def inverse_transform(self, data, seq_y, mean, std):
        return self.scaler.inverse_transform(data, seq_y, mean, std)


"""Long range dataloader for dataset elect and app flow"""


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', dataset='elect',
                 inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test']
        self.flag = flag

        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.all_data, self.covariates, self.train_end = eval('preprocess_' + dataset)(preprocess_path)
        self.all_data = torch.from_numpy(self.all_data).transpose(0, 1)
        self.covariates = torch.from_numpy(self.covariates)
        self.test_start = self.train_end - self.seq_len + 1
        self.window_stride = 24
        self.seq_num = self.all_data.size(0)

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output * (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y

    def __len__(self):
        if self.flag == 'train':
            self.window_per_seq = (self.train_end - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num
        else:
            self.window_per_seq = (self.all_data.size(
                1) - self.test_start - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num

    def __getitem__(self, index):
        seq_idx = index // self.window_per_seq
        window_idx = index % self.window_per_seq

        if self.flag == 'train':
            s_begin = window_idx * self.window_stride
        else:
            s_begin = self.test_start + window_idx * self.window_stride

        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.all_data[seq_idx, s_begin:s_end].clone()
        seq_y = self.all_data[seq_idx, r_begin:r_end].clone()
        mean, std = self.fit(seq_x)
        if mean > 0:
            seq_x = seq_x / (mean + 1)
            seq_y = seq_y / (mean + 1)

        if len(self.covariates.size()) == 2:
            seq_x_mark = self.covariates[s_begin:s_end]
            seq_x_mark[:, -1] = int(seq_idx)
            seq_y_mark = self.covariates[r_begin:r_end]
            seq_y_mark[:, -1] = int(seq_idx)
        else:
            seq_x_mark = self.covariates[s_begin:s_end, seq_idx]
            seq_x_mark[:, -1] = int(seq_idx)
            seq_y_mark = self.covariates[r_begin:r_end, seq_idx]
            seq_y_mark[:, -1] = int(seq_idx)

        return seq_x.unsqueeze(1), seq_y.unsqueeze(1), seq_x_mark, seq_y_mark, mean, std


"""Long range dataloader for synthetic dataset"""


class Dataset_Synthetic(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='synthetic.npy', dataset='synthetic',
                 inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test']
        self.flag = flag
        self.inverse = inverse

        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.all_data = np.load(preprocess_path)
        self.all_data = torch.from_numpy(self.all_data)
        self.all_data, self.covariates = self.all_data[:, :, 0], self.all_data[:, :, 1:]
        self.seq_num = self.all_data.size(0)
        print(self.seq_num)
        self.window_stride = 24
        window_per_seq = (self.all_data.shape[1] - self.seq_len - self.pred_len) / self.window_stride
        self.train_end = self.seq_len + self.pred_len + int(0.9 * window_per_seq) * self.window_stride
        self.test_start = self.train_end - self.seq_len + 1

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output * (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y

    def __len__(self):
        if self.flag == 'train':
            self.window_per_seq = (self.train_end - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num
        else:
            self.window_per_seq = (self.all_data.size(
                1) - self.test_start - self.seq_len - self.pred_len) // self.window_stride
            return self.window_per_seq * self.seq_num

    def __getitem__(self, index):
        seq_idx = index // self.window_per_seq
        window_idx = index % self.window_per_seq

        if self.flag == 'train':
            s_begin = window_idx * self.window_stride
        else:
            s_begin = self.test_start + window_idx * self.window_stride

        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.all_data[seq_idx, s_begin:s_end].clone()
        seq_y = self.all_data[seq_idx, r_begin:r_end].clone()

        mean, std = self.fit(seq_x)
        if mean > 0:
            seq_x = seq_x / (mean + 1)
            seq_y = seq_y / (mean + 1)

        seq_x_mark = self.covariates[seq_idx, s_begin:s_end]
        seq_y_mark = self.covariates[seq_idx, r_begin:r_end]

        return seq_x.unsqueeze(1), seq_y.unsqueeze(1), seq_x_mark, seq_y_mark, mean, std


def get_all_v(train_data, train_end, seq_len, pred_len, window_stride, type):
    """Get the normalization parameters of each sequence"""
    seq_num = train_data.size(0)
    window_per_seq = (train_end - seq_len - pred_len) // window_stride
    window_number = seq_num * window_per_seq

    v = torch.zeros(window_number, dtype=torch.float64)
    for index in range(window_number):
        seq_idx = index // window_per_seq
        window_idx = index % window_per_seq

        s_begin = window_idx * window_stride
        s_end = s_begin + seq_len

        seq_x = train_data[seq_idx, s_begin:s_end].clone()
        if type == 'mean':
            mean = seq_x.mean()
            v[index] = mean + 1
        else:
            std = seq_x.std()
            v[index] = std

    return v


def gen_covariates(times, num_covariates):
    """Get covariates"""
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.weekday() / 7
        covariates[i, 1] = input_time.hour / 24
        covariates[i, 2] = input_time.month / 12

    return covariates


def preprocess_elect(csv_path):
    """preprocess the elect dataset for long range forecasting"""
    num_covariates = 4
    train_start = '2011-01-01 00:00:00'
    train_end = '2014-04-01 23:00:00'
    test_start = '2014-04-01 00:00:00'
    test_end = '2014-09-07 23:00:00'

    data_frame = pd.read_csv(csv_path, sep=";", index_col=0, parse_dates=True, decimal=',')
    data_frame = data_frame.resample('1H', label='left', closed='right').sum()[train_start:test_end]
    data_frame.fillna(0, inplace=True)

    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)
    all_data = data_frame[train_start:test_end].values
    data_start = (all_data != 0).argmax(axis=0)  # find first nonzero value in each time series
    train_end = len(data_frame[train_start:train_end].values)

    all_data = all_data[:, data_start < 10000]
    data_start = data_start[data_start < 10000]
    split_start = data_start.max()
    all_data = all_data[split_start:]
    covariates = covariates[split_start:]
    train_end = train_end - split_start

    return all_data.astype(np.float32), covariates.astype(np.float32), train_end


def preprocess_flow(csv_path):
    """preprocess the app flow dataset for long range forecasting"""
    data_frame = pd.read_csv(csv_path, header=0, parse_dates=True)  # names=['app_name', 'zone', 'time', 'value']
    data_frame = data_frame.drop(data_frame.columns[0], axis=1)
    grouped_data = list(data_frame.groupby(["app_name", "zone"]))
    # covariates = gen_covariates(data_frame.index, 3)
    all_data = []
    min_length = 10000
    for i in range(len(grouped_data)):
        single_df = grouped_data[i][1].drop(labels=['app_name', 'zone'], axis=1).sort_values(by="time", ascending=True)
        times = pd.to_datetime(single_df.time)
        single_df['weekday'] = times.dt.dayofweek / 7
        single_df['hour'] = times.dt.hour / 24
        single_df['month'] = times.dt.month / 12
        temp_data = single_df.values[:, 1:]
        if (temp_data[:, 0] == 0).sum() / len(temp_data) > 0.2 or len(temp_data) < 3000:
            continue

        if len(temp_data) < min_length:
            min_length = len(temp_data)

        all_data.append(temp_data)

    all_data = np.array([data[len(data) - min_length:, :] for data in all_data]).transpose(1, 0, 2).astype(np.float32)
    train_end = min(int(0.8 * min_length), min_length - 1000)
    covariates = all_data.copy()
    covariates[:, :, :-1] = covariates[:, :, 1:]

    return all_data[:, :, 0], covariates, train_end


def gen_covariates2(times, num_covariates):
    """Get covariates"""
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.day
        covariates[i, 3] = input_time.month
    for i in range(1, num_covariates):
        covariates[:, i] = zscore(covariates[:, i])
    return covariates[:, :num_covariates]


"""Single step dataloader"""


def split(split_start, label, cov, pred_length):
    all_data = []
    for batch_idx in range(len(label)):
        batch_label = label[batch_idx]
        for i in range(pred_length):
            single_data = batch_label[i:(split_start + i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[batch_idx, i:(split_start + i), :].clone()
            temp_data = [single_data, single_cov]
            single_data = torch.cat(temp_data, dim=1)
            all_data.append(single_data)
    data = torch.stack(all_data, dim=0)
    label = label[:, -pred_length:].reshape(pred_length * len(label))

    return data, label


"""Single step training dataloader for the electricity dataset"""


class electTrainDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length, batch_size):
        self.data = torch.from_numpy(np.load(os.path.join(data_path, f'train_data_{data_name}.npy')))

        # Resample windows according to the average amplitude
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        weights = torch.as_tensor(np.abs(v[:, 0]) / np.sum(np.abs(v[:, 0])), dtype=torch.double)
        num_samples = weights.size(0)
        sample_index = torch.multinomial(weights, num_samples, True)
        self.data = self.data[sample_index]

        self.label = torch.from_numpy(np.load(os.path.join(data_path, f'train_label_{data_name}.npy')))
        self.label = self.label[sample_index]

        self.train_len = len(self.data) // batch_size
        self.pred_length = predict_length
        self.batch_size = batch_size

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if (index + 1) <= self.train_len:
            all_data = self.data[index * self.batch_size:(index + 1) * self.batch_size].clone()
            label = self.label[index * self.batch_size:(index + 1) * self.batch_size].clone()
        else:
            all_data = self.data[index * self.batch_size:].clone()
            label = self.label[index * self.batch_size:].clone()

        cov = all_data[:, :, 2:]

        split_start = len(label[0]) - self.pred_length + 1
        data, label = split(split_start, label, cov, self.pred_length)

        return data, label


"""Single step testing dataloader for the electricity dataset"""


class electTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'test_label_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        self.pred_length = predict_length

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 2:]
        label = torch.from_numpy(self.label[index].copy())
        v = float(self.v[index][0])
        if v > 0:
            data = label / v
        else:
            data = label

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start + i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start + i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:]

        return all_data, label, v


"""Single step training dataloader for the app flow dataset"""


class flowTrainDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length, batch_size):
        self.data = torch.from_numpy(np.load(os.path.join(data_path, f'train_data_{data_name}.npy')))

        # Resample windows according to the average amplitude
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        weights = torch.as_tensor(np.abs(v) / np.sum(np.abs(v)), dtype=torch.double)
        num_samples = weights.size(0)
        sample_index = torch.multinomial(weights, num_samples, True)
        self.data = self.data[sample_index]

        self.label = self.data[:, :, 0]

        self.train_len = len(self.data) // batch_size
        self.pred_length = predict_length
        self.batch_size = batch_size

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if (index + 1) <= self.train_len:
            all_data = self.data[index * self.batch_size:(index + 1) * self.batch_size].clone()
            label = self.label[index * self.batch_size:(index + 1) * self.batch_size].clone()
        else:
            all_data = self.data[index * self.batch_size:].clone()
            label = self.label[index * self.batch_size:].clone()

        cov = all_data[:, :, 1:]

        split_start = len(label[0]) - self.pred_length + 1
        data, label = split(split_start, label, cov, self.pred_length)

        return data, label


"""Single step testing dataloader for the all flow dataset"""


class flowTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.label = self.data
        self.test_len = self.data.shape[0]
        self.pred_length = predict_length

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 1:]
        data = all_data[:, 0]
        label = torch.from_numpy(self.label[index, :, 0].copy())
        v = float(self.v[index])

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start + i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start + i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:] * v

        return all_data, label, v


"""Single step training dataloader for the wind dataset"""


class windTrainDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length, batch_size):
        self.data = torch.from_numpy(np.load(os.path.join(data_path, f'train_data_{data_name}.npy')))

        # Resample windows according to the average amplitude
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        weights = torch.as_tensor(np.abs(v) / np.sum(np.abs(v)), dtype=torch.double)
        num_samples = weights.size(0)
        sample_index = torch.multinomial(weights, num_samples, True)
        self.data = self.data[sample_index]

        self.train_len = len(self.data) // batch_size
        self.pred_length = predict_length
        self.batch_size = batch_size

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if (index + 1) <= self.train_len:
            all_data = self.data[index * self.batch_size:(index + 1) * self.batch_size].clone()
        else:
            all_data = self.data[index * self.batch_size:].clone()

        cov = all_data[:, :, 1:]
        label = all_data[:, :, 0]

        split_start = len(label[0]) - self.pred_length + 1
        data, label = split(split_start, label, cov, self.pred_length)

        return data, label


"""Single step testing dataloader for the wind dataset"""


class windTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length):
        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        self.pred_length = predict_length

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 1:]
        data = all_data[:, 0]
        v = float(self.v[index])
        label = data * v

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start + i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start + i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:]

        return all_data, label, v


"""Single step training dataloader for the camel dataset"""


class camelTrainDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length, batch_size):
        self.data = torch.from_numpy(np.load(os.path.join(data_path, f'train_data_{data_name}.npy')))

        # Resample windows according to the average amplitude
        v = np.load(os.path.join(data_path, f'train_v_{data_name}.npy'))
        weights = torch.as_tensor(np.abs(v[:, 0]) / np.sum(np.abs(v[:, 0])), dtype=torch.double)
        num_samples = weights.size(0)
        sample_index = torch.multinomial(weights, num_samples, True)
        self.data = self.data[sample_index]

        self.label = torch.from_numpy(np.load(os.path.join(data_path, f'train_label_{data_name}.npy')))
        self.label = self.label[sample_index]

        self.train_len = len(self.data) // batch_size
        self.pred_length = predict_length
        self.batch_size = batch_size

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        if (index + 1) <= self.train_len:
            all_data = self.data[index * self.batch_size:(index + 1) * self.batch_size].clone()
            label = self.label[index * self.batch_size:(index + 1) * self.batch_size].clone()
        else:
            all_data = self.data[index * self.batch_size:].clone()
            label = self.label[index * self.batch_size:].clone()

        cov = all_data[:, :, 2:]

        split_start = len(label[0]) - self.pred_length + 1
        data, label = split(split_start, label, cov, self.pred_length)

        return data, label


"""Single step testing dataloader for the camel dataset"""


class camelTestDataset(Dataset):
    def __init__(self, data_path, data_name, predict_length):

        self.data = np.load(os.path.join(data_path, f'test_data_{data_name}.npy'))
        self.v = np.load(os.path.join(data_path, f'test_v_{data_name}.npy'))
        self.label = np.load(os.path.join(data_path, f'test_label_{data_name}.npy'))
        self.test_len = self.data.shape[0]
        self.pred_length = predict_length

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        all_data = torch.from_numpy(self.data[index].copy())
        cov = all_data[:, 2:]
        label = torch.from_numpy(self.label[index].copy())
        v = float(self.v[index][0])
        if v > 0:
            data = label / v
        else:
            data = label

        split_start = len(label) - self.pred_length + 1
        all_data = []
        for i in range(self.pred_length):
            single_data = data[i:(split_start + i)].clone().unsqueeze(1)
            single_data[-1] = -1
            single_cov = cov[i:(split_start + i), :].clone()
            single_data = torch.cat([single_data, single_cov], dim=1)
            all_data.append(single_data)
        all_data = torch.stack(all_data, dim=0)
        label = label[-self.pred_length:]

        return all_data, label, v
