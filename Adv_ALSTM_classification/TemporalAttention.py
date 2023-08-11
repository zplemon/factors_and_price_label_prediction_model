import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalAttention(nn.Module):
    def __init__(self,fin,fout):
        super(TemporalAttention, self).__init__()
        self.fin = fin
        self.fout = fout

        self.w = nn.Parameter(torch.Tensor(self.fin,self.fout), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.fin)
        for weight in self.w:
            nn.init.uniform_(weight, -stdv, stdv)
        # nn.init.xavier_uniform_(self.w, gain=1.414)

    def forward(self,h):
        x = h
        alpha = torch.matmul(h,self.w)
        alpha = F.softmax(torch.tanh(alpha),1)
        x = torch.einsum('ijk,ijm->ikm',alpha,x)
        return torch.squeeze(x,1)
