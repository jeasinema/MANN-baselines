"""
A collection of key-value-based memory.
Note: all memories must be reset before use.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from memory import ValueMemory


class KeyValueMemory(ValueMemory):
    """ValueMemory with:

        -K-V arch
        used by [so far I don't know]
    """
    def __init__(self, mem_size, value_size, batch_size, key_size):
        super(KeyValueMemory, self).__init__(
            mem_size,
            value_size,
            batch_size,
        )
        self.key_size = key_size
        self.register_buffer('memory_key', torch.FloatTensor(batch_size, mem_size, key_size))

    @property
    def shape(self, with_batch_dim=False):
        if with_batch_dim:
            return self.batch_size, self.mem_size, self.value_size, self.key_size
        else:
            return self.mem_size, self.value_size, self.key_size

    def reset(self):
        stdev = 1 / (np.sqrt(self.mem_size + self.key_size))
        nn.init.uniform_(self.memory_key, -stdev, stdev)
        super(KeyValueMemory, self).reset()

    def similarity(self, k, topk=-1):
        """
        :output similarity: shape (B, self.mem_size)

        The output should be treated unnormalized.

        :param k: shape (B, self.key_size)
        """
        B = k.size(0)
        sim = F.cosine_similarity(self.memory_key[:B] + LOG_EPS, k.unsqueeze(1) + LOG_EPS, dim=-1)
        if topk != -1:
            # Zero-out the non-topk
            ind = torch.topk(sim, self.mem_size - topk, dim=-1, largest=False)[1]
            sim.scatter_(1, ind, 0)
        return sim

    def write(self, w, k, v):
        """
        :param w: shape (B, self.mem_size)
        :param k: shape (B, self.key_size)
        :param v: shape (B, self.value_size)
        """
        B = w.size(0)
        write_k = torch.matmul(w.unsqueeze(-1), k.unsqueeze(1))
        write_v = torch.matmul(w.unsqueeze(-1), v.unsqueeze(1))
        self.memory_key[:B] = self.memory_key[:B] + write_k
        self.memory[:B] = self.memory[:B] + write_v


class DNDMemory(KeyValueMemory):
    """KeyValueMemory with:

        -appending-based write
        -overwrite the least recently r/w entry
        used by DND, Memory Networks
    """
    def __init__(self, *args, **kwargs):
        super(DNDMemory, self).__init__(*args, **kwargs)
        self.gamma = 0.9
        self.register_buffer('mem_usage', torch.FloatTensor(self.batch_size, self.mem_size).fill_(0))

    def read(self, w):
        B = w.size(0)
        self.mem_usage[:B] *= self.gamma
        self.mem_usage[:B] += w
        return super(DNDMemory, self).read(w)

    def write(self, w, k, v):
        """
        :param w: shape (B, self.mem_size)
        :param k: shape (B, self.key_size)
        :param v: shape (B, self.value_size)
        """
        B = w.size(0)
        write_k = torch.matmul(w.unsqueeze(-1), k.unsqueeze(1))
        write_v = torch.matmul(w.unsqueeze(-1), v.unsqueeze(1))
        self.memory_key[:B] = write_k
        self.memory[:B] = write_v

    def write_least_used(self, k, v):
        """
        :output: weight for least-used overwriting shape (B, self.mem_size)

        :param k: shape (B, self.key_size)
        :param v: shape (B, self.value_size)
        """
        B = k.size(0)
        ind = torch.topk(self.mem_usage[:B], 1, -1, largest=False)[1]
        w = torch.zeros(B, self.mem_size).to(k)
        w.scatter_(1, ind, 1)
        self.write(w, k, v)
        self.mem_usage[:B] *= self.gamma
        self.mem_usage[:B] += w
        return w


class MRAMemory(DNDMemory):
    """DNDMemory with:

        -L2+topk based similarity (TODO:jxma)
        used by MRA
    """
    def similarity(self, k, topk=-1):
        super(MRAMemory, self).similarity(k, topk)
