# -*- coding:utf-8 -*-

import copy
import os
import sys

import pandas as pd

from get_data import setup_seed, device
from adv_function import get_adv

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain

import torch
from scipy.interpolate import make_interp_spline
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import ALSTM, BiLSTM
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score

setup_seed(20)


def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.CrossEntropyLoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            final_map = nn.Linear(model.num_directions * 2 * args.hidden_size, 5, bias=True)
            seq = seq.to(args.device)
            label = label.to(args.device)
            e_s = model(seq)
            y_s = final_map(e_s)
            # no adv is applied to validation set
            # r_adv = get_adv(seq, label, final_map, model, 1e-2, loss_function)
            # e_adv = e_s + r_adv
            # y_adv = final_map(e_adv)
            loss_1 = loss_function(y_s, label.flatten())
            # loss_2 = loss_function(y_adv, label.flatten())
            # loss = loss_1.item() + args.beta * loss_2.item()
            # val_loss.append(loss)
            val_loss.append(loss_1.item())
    return np.mean(val_loss)


def train(args, Dtr, Val, path):
    if args.bidirectional:
        model = BiLSTM(args).to(device)
    else:
        model = ALSTM(args).to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            final_map = nn.Linear(model.num_directions * 2 * args.hidden_size, 5, bias=True)
            seq = seq.to(device)
            label = label.to(device)
            e_s = model(seq)
            y_s = final_map(e_s)
            if model.adv:
                r_adv = get_adv(seq, label, final_map, model, 1e-2, loss_function)
                e_adv = e_s + r_adv
                y_adv = final_map(e_adv)
            optimizer.zero_grad()
            loss_1 = loss_function(y_s, label.flatten())
            if model.adv:
                loss_2 = loss_function(y_adv, label.flatten())
                loss = loss_1.item() + args.beta*loss_2.item()
                train_loss.append(loss)
            else:
                train_loss.append(loss_1.item())
            loss_1.backward()
            optimizer.step()
        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model': best_model.state_dict()}
    torch.save(state, path)


def test(args, Dte, path):
    pred = []
    y = []
    print('loading models...')
    if args.bidirectional:
        model = BiLSTM(args).to(device)
    else:
        model = ALSTM(args).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            final_map = nn.Linear(model.num_directions * 2 * args.hidden_size, 5, bias=True)
            e_s = model(seq)
            y_s = final_map(e_s)
            _, _pred = y_s.max(dim=1)
            pred.extend(_pred.cpu().numpy().tolist())

    y, pred = np.array(y), np.array(pred)
    res = pd.DataFrame({'y': y, 'pred': pred})
    res.to_csv("res_49_adv_5_class_1.csv")
    print('acc:', accuracy_score(y, pred))
