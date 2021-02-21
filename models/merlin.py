import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import haste_pytorch as haste

from memory import *
from models import SimpleNTM


class RLMEM(SimpleNTM):
    """`RL-MEM` in MERLIN (MERLIN w/o variational loss)

        -value-based memory
        -simplified reading
        -overwritting-based write
        https://arxiv.org/pdf/1803.10760.pdf
    """
    def __init__(
        self,
        # Basics
        encoder,
        encoder_output_size,
        model_output_size,
        batch_size,
        # Memory
        mem_size=10,
        mem_value_size=256,
        # Controller
        controller='lstm',
        controller_hidden_units=None,
        controller_output_size=128,
        # R/W head
        num_read_heads=1,
        num_write_heads=1,
        gamma=1.0):
        self.gamma = 1.0
        super(RLMEM, self).__init__(
            # Basics
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=MERLINMemory,
            mem_size=mem_size,
            mem_value_size=mem_value_size*2,
            # Controller
            controller=controller,
            controller_hidden_units=controller_hidden_units,
            controller_output_size=controller_output_size,
            # R/W head
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            read_length=[mem_value_size*2, 1],
            write_length=[mem_value_size]
        )

    def _address(self, k, β):
        """NTM Addressing (according to section 3.3).
        :output weight: shape (B, self.memory.mem_size)

        :param k: shape (B, .) The key vector.
        :param β: shape (B, .) The key strength (focus).
        """
        # Handle Activations
        β = F.softplus(β)

        # Content focus
        sim = self.memory.similarity(k)
        w = F.softmax(β * sim, dim=1)

        return w

    def _read(self, x, prev_rhead_state):
        """
        :output read value: shape list of (B, self.memory.value_size), length == len(self.read_heads)
        :output r head weight: shape list of (B, self.memory.mem_size), length == len(self.read_heads)

        :param x: shape (B, self.controller_output_size)
        :param prev_rhead_state: list of (B, self.memory.mem_size), length == len(self.read_heads)
        """
        reads = []
        ws = []
        for rhead in self.read_heads:
            o = rhead(x)
            k, β = split_cols(o, self.read_length)
            w = self._address(k, β)
            r = self.memory.read(w)
            ws.append(w)
            reads.append(r)
        return reads, ws

    def _write(self, x, prev_whead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.controller_output_size)
        :param prev_whead_state: list of ((B, self.memory.mem_size) * 2), length == len(self.write_heads)
        """
        ws = []
        for whead, w_prev in zip(self.write_heads, prev_whead_state):
            if type(w_prev) is tuple:
                w_ret_prev, w_wr_prev = w_prev
            # prev_whead_state is initialized
            else:
                w_ret_prev, w_wr_prev = w_prev, w_prev.clone()
            v = whead(x)
            w_ret = self.gamma * w_ret_prev + (1 - self.gamma) * w_wr_prev
            padding_zero = torch.zeros_like(v)
            self.memory.write(w_ret, torch.cat([padding_zero, v], dim=-1))
            w_wr = self.memory.write_least_used(torch.cat((v, padding_zero), dim=-1))
            ws.append((w_ret, w_wr))
        return ws
