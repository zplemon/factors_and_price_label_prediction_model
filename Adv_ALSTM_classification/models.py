# -*- coding:utf-8 -*-

from torch import nn
import torch
from TemporalAttention import TemporalAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ALSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.E = args.E
        self.num_directions = 1
        self.adv = args.adv
        self.attention = TemporalAttention(self.num_directions * self.hidden_size, 1)
        self.lstm = nn.LSTM(self.E, self.hidden_size, self.num_layers, batch_first=True)
        self.mapping_layer = nn.Linear(self.input_size, self.E, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        # print(input_seq.shape)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        input_seq = input_seq.flatten(start_dim = 0, end_dim = 1)
        mapping_feature = self.tanh(self.mapping_layer(input_seq))
        mapping_feature = mapping_feature.reshape(batch_size, seq_len, -1)
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # print(mapping_feature.shape)

        output, (hn, cn) = self.lstm(mapping_feature, (h_0, c_0))
        hn = hn.squeeze(0)

        a_s = self.attention(output)
        h_T = output[:, -1, :]  # last output of the hidden layers of the LSTM block
        # print(a_s.shape)
        # print(h_T.shape)
        es = torch.cat([a_s, h_T], dim=1)
        return es


class BiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.E = args.E
        self.adv = args.adv
        self.num_directions = 2
        self.attention = TemporalAttention(self.num_directions * self.hidden_size, 1)
        self.lstm = nn.LSTM(self.E, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.mapping_layer = nn.Linear(self.input_size, self.E, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        input_seq = input_seq.flatten(start_dim=0, end_dim=1)
        mapping_feature = self.tanh(self.mapping_layer(input_seq))
        mapping_feature = mapping_feature.reshape(batch_size, seq_len, -1)
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)

        output, (hn, cn) = self.lstm(mapping_feature, (h_0, c_0))
        hn = hn.squeeze(0)

        a_s = self.attention(output)
        h_T = output[:, -1, :]  # last output of the hidden layers of the LSTM block
        # print(a_s.shape)
        # print(h_T.shape)
        es = torch.cat([a_s, h_T], dim=1)
        return es
