"""
A collection of value-based memory.
Note: all memories must be reset before use.
"""
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

    def __init__(self, mem_size, value_size, batch_size):
        super(ValueMemory, self).__init__()
        self.mem_size = mem_size
        self.value_size = value_size
        self.batch_size = batch_size

        # Mem create and init
        self.register_buffer('memory', torch.FloatTensor(self.batch_size, self.mem_size, self.value_size))

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
        TODO: topk + sparse matrix product
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

    def clear(self, w):
        """
        :param w: shape (B, self.mem_size) should be mulit-hot tensor (0 or 1)
        """
        assert ((w == 0) + (w == 1)).all()
        B = w.size(0)
        self.memory[:B] *= w.unsqueeze(-1)

    def write(self, w, v, clear_before_write=False):
        """
        :param w: shape (B, self.mem_size)
        :param v: shape (B, self.value_size)
        """
        B = w.size(0)
        if clear_before_write:
            self.clear(w)
        write_v = torch.matmul(w.unsqueeze(-1), v.unsqueeze(1))
        self.memory[:B] += write_v

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


class AppendingMemory(ValueMemory):
    """ValueMemory with:

        -appending-based write
        -overwrite the least recently r/w entry
    """
    def __init__(self, *args, **kwargs):
        super(AppendingMemory, self).__init__(*args, **kwargs)
        self.gamma = 0.9
        self.register_buffer('mem_usage', torch.FloatTensor(self.batch_size, self.mem_size).fill_(0))

    def read(self, w):
        """
        :output read value: shape (B, self.value_size)

        :param w: shape (B, self.mem_size)
        """
        B = w.size(0)
        self.mem_usage[:B] *= self.gamma
        self.mem_usage[:B] += w
        return super(AppendingMemory, self).read(w)

    def write(self, w, v, clear_before_write=False):
        """
        :param w: shape (B, self.mem_size)
        :param v: shape (B, self.value_size)
        """
        B = w.size(0)
        self.mem_usage[:B] *= self.gamma
        self.mem_usage[:B] += w
        super(AppendingMemory, self).write(w, v, clear_before_write=clear_before_write)

    def write_least_used(self, v):
        """
        :output: weight for least-used overwriting shape (B, self.mem_size)

        :param v: shape (B, self.value_size)
        """
        B = v.size(0)
        ind = torch.topk(self.mem_usage[:B], 1, -1, largest=False)[1]
        w = torch.zeros(B, self.mem_size).to(v)
        w.scatter_(1, ind, 1)
        self.write(w, v, clear_before_write=True)
        return w


class MERLINMemory(ValueMemory):
    """ValueMemory with:

        -usage counting during reading (for overwritting)
        used by RLMEM, MERLIN

        Note: we use replace-based write here, which is slightly different from
            the original paper(https://arxiv.org/pdf/1803.10760.pdf).
    """
    def __init__(self, *args, **kwargs):
        super(MERLINMemory, self).__init__(*args, **kwargs)
        self.register_buffer('mem_usage', torch.FloatTensor(self.batch_size, self.mem_size).fill_(0))

    def read(self, w):
        """
        :output read value: shape (B, self.value_size)

        :param w: shape (B, self.mem_size)
        """
        B = w.size(0)
        self.mem_usage[:B] += w
        return super(MERLINMemory, self).read(w)

    def write_least_used(self, v):
        """
        :output: weight for least-used overwriting shape (B, self.mem_size)

        :param v: shape (B, self.value_size)
        """
        B = v.size(0)
        ind = torch.topk(self.mem_usage[:B], 1, -1, largest=False)[1]
        w = torch.zeros(B, self.mem_size).to(v)
        w.scatter_(1, ind, 1)
        self.write(w, v, clear_before_write=True)
        return w


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
