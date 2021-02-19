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
        self.key_size = key_size
        # Key mem create and init
        self.register_buffer('memory_key', torch.FloatTensor(self.batch_size, self.mem_size, self.key_size))
        super(KeyValueMemory, self).__init__(
            mem_size,
            value_size,
            batch_size,
        )

    @property
    def shape(self, with_batch_dim=False):
        if with_batch_dim:
            return self.batch_size, self.mem_size, self.value_size, self.key_size
        else:
            return self.mem_size, self.value_size, self.key_size

    def reset(self):
        stdev = 1 / (np.sqrt(self.mem_size + self.key_size))
        nn.init.uniform_(self.memory_key, -stdev, stdev)
        super().reset()

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
        used by DND, Memory Networks
    """
    def __init__(self, init_mem_size, value_size, batch_size, key_size):
        self.init_mem_size = init_mem_size
        self.used_mem = 0
        super(DNDMemory, self).__init__(init_mem_size, key_size, value_size, batch_size)

    def reset(self):
        self.mem_size = self.init_mem_size
        # Free CPU/GPU memory
        del self.memory
        del self.memory_key
        torch.cuda.empty_cache()
        self.register_buffer('memory', torch.FloatTensor(self.batch_size, self.init_mem_size, self.value_size))
        self.register_buffer('memory_key', torch.FloatTensor(self.batch_size, self.init_mem_size, self.key_size))
        stdev = 1 / (np.sqrt(self.init_mem_size + self.value_size))
        stdev_key = 1 / (np.sqrt(self.init_mem_size + self.key_size))
        nn.init.uniform_(self.memory, -stdev, stdev)
        nn.init.uniform_(self.memory_key, -stdev_key, stdev_key)

    def write(self, k, v):
        """
        TODO (jxma): better re-allocation strategy

        :param k: shape (B, self.key_size)
        :param v: shape (B, self.value_size)
        """
        B = k.size(0)
        self.used_mem += 1
        if self.used_mem > self.mem_size:
            self.mem_size += 1
            self.memory.resize_(self.batch_size, self.mem_size, self.value_size)
            self.memory_key.resize_(self.batch_size, self.mem_size, self.key_size)
            self.memory[:B, -1, :] = v
            self.memory_key[:B, -1, :] = k
        else:
            self.memory[:B, self.used_mem-1, :] = v
            self.memory_key[:B, self.used_mem-1, :] = k


class MRAMemory(DNDMemory):
    """DNDMemory with:

        -L2+topk based similarity (TODO:jxma)
        used by MRA
    """
    def similarity(self, k, topk=-1):
        super(MRAMemory, self).similarity(k, topk)
