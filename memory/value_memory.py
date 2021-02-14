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
        self.B = batch_size

        # Mem create and init
        if jumpy_bp:
            self.register_buffer('memory', torch.FloatTensor(self.B, self.mem_size, self.value_size))
        else:
            self.register_parameter('memory', nn.Parameter(torch.FloatTensor(self.B, self.mem_size, self.value_size)))
        stdev = 1 / (np.sqrt(self.mem_size + self.value_size))
        nn.init.uniform_(self.memory, -stdev, stdev)

    @property
    def shape(self, with_batch_dim=False):
        if with_batch_dim:
            return self.B, self.mem_size, self.value_size
        else:
            return self.mem_size, self.value_size

    def reset(self):
        stdev = 1 / (np.sqrt(self.mem_size + self.value_size))
        nn.init.uniform_(self.memory, -stdev, stdev)

    def similarity(self, v, topk=-1):
        """
        Output: shape (B, self.mem_size)

        The output should be treated unnormalized.

        :param v: shape (B, self.value_size)
        """
        sim = F.cosine_similarity(self.memory + LOG_EPS, v.unsqueeze(1) + LOG_EPS, dim=-1)           
        if topk != -1:
            # Zero-out the non-topk
            ind = torch.topk(sim, self.mem_size - topk, dim=-1, largest=False)[1]
            sim.scatter_(1, ind, 0)
        return sim

    def write(self, w, v):
        """
        Output: None

        :param w: shape (B, self.mem_size)
        :param v: shape (B, self.value_size)
        """
        write_v = torch.matmul(w.unsqueeze(-1), v.unsqueeze(1))
        self.memory = self.memory + write_v

    def read(self, w):
        """
        Output: shape (B, self.value_size)

        :param w: shape (B, self.mem_size)
        """
        return torch.matmul(w.unsqueeze(1), self.memory).unsqueeze(1)


class NTMMemory(ValueMemory):
    """ValueMemory with:
        
        -add/del-based write
        used by NTM
    """
    def write(self, w, add_v, del_v):
        """
        Output: None

        :param w: shape (B, self.mem_size)
        :param add_v: shape (B, self.value_size)
        :param del_v: shape (B, self.value_size)
        """
    write_add_v = torch.matmul(w.unsqueeze(-1), add_v.unsqueeze(1))
    write_del_v = torch.matmul(w.unsqueeze(-1), del_v.unsqueeze(1))
    self.memory = self.memory * (1 - write_del_v) + write_add_v 


class MERLINMemory(ValueMemory):
    """NTMMemory with:

        -appending-based write
        used by RL-MEM, MERLIN
    """
    def __init__(self, init_mem_size, value_size, batch_size, jumpy_bp=True):
        self.init_mem_size = init_mem_size
        self.used_mem = 0
        super(MERLINMemory, self).__init__(init_mem_size, value_size, batch_size, jumpy_bp)

    def reset(self):
        self.mem_size = self.init_mem_size
        # Free CPU/GPU memory
        del self.memory
        if self.jumpy_bp:
            self.register_buffer('memory', torch.FloatTensor(self.B, self.init_mem_size, self.value_size))
        else:
            self.register_parameter('memory', nn.Parameter(torch.FloatTensor(self.B, self.init_mem_size, self.value_size)))
        stdev = 1 / (np.sqrt(self.init_mem_size + self.value_size))
        nn.init.uniform_(self.memory, -stdev, stdev)

    def write(self, v):
        """
        Output: None

        TODO (jxma): better re-allocation strategy

        :param v: shape (B, self.value_size)
        """
        self.used_mem += 1
        if self.used_mem > self.mem_size:
            self.mem_size += 1
            self.memory.resize_(self.B, self.mem_size, self.value_size)
            self.memory[:, -1, :] = v 
        else:
            self.memory[:, self.used_mem-1, :] = v


class DNCMemory(NTMMemory):
    """NTMMemory with:
        
        -learnable allocation
        used by DNC    
    """
    raise NotImplementedError


class RTSMemory(DNDMemory):
    """MERLINMemory with:

        -all-reduce read
        used by RTS
    """
    def read(self):
        """
        Output: shape (self.B, self.mem_size, self.value_size)
        """
        return self.memory