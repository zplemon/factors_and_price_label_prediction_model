# -*- coding:utf-8 -*-

import os
import sys

from args import args_parser
from get_data import nn_seq
from util import train, test

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = 'models/model.pkl'

if __name__ == '__main__':
    args = args_parser()
    Dtr, Val, Dte = nn_seq(seq_len=args.seq_len, B=args.batch_size,
                                                      num=args.output_size)
    train(args, Dtr, Val, LSTM_PATH)
    test(args, Dte, LSTM_PATH)