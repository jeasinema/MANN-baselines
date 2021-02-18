import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class ValueMemory(nn.Module):
    """
        Canonial value-based memory.
        used by MANN(http://proceedings.mlr.press/v48/santoro16.pdf)
    """

    def __init__(self, mem_size, value_size, batch_size, jumpy_bp=True):
        super(ValueMemory, self).__init__()
        self.mem_size = mem_size
        self.value_size = value_size
        self.batch_size = batch_size

        # Mem create and init
        if jumpy_bp:
            self.register_buffer('memory', torch.FloatTensor(self.batch_size, self.mem_size, self.value_size))
        else:
            self.register_parameter('memory', nn.Parameter(torch.FloatTensor(self.batch_size, self.mem_size, self.value_size)))

        self.reset()

    @property
    def shape(self, with_batch_dim=False):
        if with_batch_dim:
            return self.batch_size, self.mem_size, self.value_size
        else:
            return self.mem_size, self.value_size

    def reset(self):
        stdev = 1 / (np.sqrt(self.mem_size + self.value_size))
        nn.init.uniform_(self.memory, -stdev, stdev)

    def similarity(self, v, topk=-1):
        """
        :output similarity: shape (B, self.mem_size)
        The output should be treated unnormalized.

        :param v: shape (B, self.value_size)
        """
        B = v.size(0)
        sim = F.cosine_similarity(self.memory[:B] + LOG_EPS, v.unsqueeze(1) + LOG_EPS, dim=-1)
        if topk != -1:
            # Zero-out the non-topk
            ind = torch.topk(sim, self.mem_size - topk, dim=-1, largest=False)[1]
            sim.scatter_(1, ind, 0)
        return sim

    def write(self, w, v):
        """
        :param w: shape (B, self.mem_size)
        :param v: shape (B, self.value_size)
        """
        B = w.size(0)
        write_v = torch.matmul(w.unsqueeze(-1), v.unsqueeze(1))
        self.memory[:B] = self.memory[:B] + write_v

    def read(self, w):
        """
        :output read value: shape (B, self.value_size)

        :param w: shape (B, self.mem_size)
        """
        B = w.size(0)
        return torch.matmul(w.unsqueeze(1), self.memory[:B]).squeeze(1)


class NTMMemory(ValueMemory):
    """ValueMemory with:

        -add/del-based write
        used by NTM
    """
    def write(self, w, add_v, del_v):
        """
        :param w: shape (B, self.mem_size)
        :param add_v: shape (B, self.value_size)
        :param del_v: shape (B, self.value_size)
        """
        B = w.size(0)
        write_add_v = torch.matmul(w.unsqueeze(-1), add_v.unsqueeze(1))
        write_del_v = torch.matmul(w.unsqueeze(-1), del_v.unsqueeze(1))
        self.memory[:B] = self.memory[:B] * (1 - write_del_v) + write_add_v


class MERLINMemory(ValueMemory):
    """ValueMemory with:

        -usage counting (for overwritting)
        used by RLMEM, MERLIN
    """
    def __init__(self, *args, **kwargs):
        super(MERLINMemory, self).__init__(*args, **kwargs)
        self.register_buffer('mem_usage', torch.FloatTensor(self.batch_size, self.mem_size).fill_(0))

    def read(self, w):
        B = w.size(0)
        self.mem_usage[:B] += w
        return super(MERLINMemory, self).read(w)

    def write_least_used(self, x):
        """
        :output: weight for least-used overwriting shape (B, self.mem_size)
        """
        B = x.size(0)
        ind = torch.topk(self.mem_usage[:B], 1, -1, largest=False)[1]
        w = torch.zeros(B, self.mem_size).to(x)
        w.scatter_(1, ind, 1)
        self.write(w, x)
        return w


class AppendingMemory(ValueMemory):
    """ValueMemory with:

        -appending-based write
    """
    def __init__(self, init_mem_size, value_size, batch_size, jumpy_bp=True):
        self.init_mem_size = init_mem_size
        self.used_mem = 0
        super(AppendingMemory, self).__init__(init_mem_size, value_size, batch_size, jumpy_bp)

    def reset(self):
        self.mem_size = self.init_mem_size
        # Free CPU/GPU memory
        del self.memory
        torch.cuda.empty_cache()
        if self.jumpy_bp:
            self.register_buffer('memory', torch.FloatTensor(self.batch_size, self.init_mem_size, self.value_size))
        else:
            self.register_parameter('memory', nn.Parameter(torch.FloatTensor(self.batch_size, self.init_mem_size, self.value_size)))
        stdev = 1 / (np.sqrt(self.init_mem_size + self.value_size))
        nn.init.uniform_(self.memory, -stdev, stdev)

    def write(self, v):
        """
        TODO (jxma): better re-allocation strategy

        :param v: shape (B, self.value_size)
        """
        B = v.size(0)
        self.used_mem += 1
        if self.used_mem > self.mem_size:
            self.mem_size += 1
            self.memory.resize_(self.batch_size, self.mem_size, self.value_size)
            self.memory[:B, -1, :] = v
        else:
            self.memory[:B, self.used_mem-1, :] = v


class DNCMemory(NTMMemory):
    """NTMMemory with:

        -learnable allocation
        used by DNC
    """
    pass


class RTSMemory(AppendingMemory):
    """MERLINMemory with:

        -all-reduce read
        used by RTS
    """
    def read(self):
        """
        :output read value: shape (self.batch_size, self.mem_size, self.value_size)
        """
        return self.memory
