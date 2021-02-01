import torch
import torch.nn as nn

class RandomEncoding(nn.Module):

    def __init__(self, d_model, max_len=9000):
        super(RandomEncoding, self).__init__()
        re = torch.randn(max_len, d_model)
        self.register_buffer("re", re)

    def forward(self, x):
        return torch.index_select(self.re, 0, x.long())