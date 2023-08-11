# -*- coding:utf-8 -*-

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data():
    """
    :return:
    """
    path1 = 'data/data_49_511380_train.csv'
    df1 = pd.read_csv(path1, encoding='gbk', index_col=0)
    df1.fillna(method='ffill', inplace=True)
    df1 = df1[df1['month'] < 5]

    path2 = 'data/data_49_511380_test.csv'
    df2 = pd.read_csv(path2, encoding='gbk', index_col=0)
    df2.index = range(len(df2))
    df2.fillna(method='ffill', inplace=True)

    df = pd.concat([df1,df2],ignore_index=True)

    # for idx, c in enumerate(df.columns):
    #     if idx == 0:
    #         continue   # 跳过时间列
    #     df[c].fillna(df[c].mean(), inplace=True)  #
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq(seq_len, B, num):
    print('data processing...')
    # dataset_train, dataset_test = load_data()
    dataset = load_data()
    # split
    train = dataset[:int(len(dataset) * 0.7)].values
    # train = dataset_train[:int(len(dataset_train) * 0.8)].values
    # val = dataset_train[int(len(dataset_train) * 0.8):].values
    val = dataset[int(len(dataset) * 0.7):int(len(dataset) * 0.8)].values
    test = dataset[int(len(dataset) * 0.8):].values
    # test = dataset_test.values

    train_x, train_y = train[:, :-2], train[:, -2]
    val_x, val_y = val[:, :-2], val[:, -2]
    test_x, test_y = test[:, :-2], test[:, -2]
    pd.DataFrame(test).to_csv('test_set_data.csv')
    pd.DataFrame(test_x).to_csv('test_x_data.csv')
    # -2 -1 0 1 2 --> 0 1 2 3 4 + 2
    train_y = train_y + 2
    val_y = val_y + 2
    test_y = test_y + 2

    # 归一化
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    def process(x, y, batch_size, step_size, shuffle):
        seq = []
        for i in range(0, len(x) - seq_len, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                temp_x = x[j].tolist()
                train_seq.append(temp_x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(y[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.LongTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train_x, train_y, B, step_size=1, shuffle=True)
    Val = process(val_x, val_y, B, step_size=1, shuffle=True)
    Dte = process(test_x, test_y, B, step_size=num, shuffle=False)

    return Dtr, Val, Dte


# def get_mape(x, y):
#     """
#     :param x: true value
#     :param y: pred value
#     :return: mape
#     """
#     return np.mean(np.abs((x - y) / x))
