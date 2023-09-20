import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


# https://zh-v2.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html
# https://www.bilibili.com/list/1567748478?sid=358497&spm_id_from=333.999.0.0&desc=1&oid=377288232&bvid=BV19o4y1m7mo&p=2
# @save
class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
