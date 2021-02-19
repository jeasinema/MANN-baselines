"""
A collection of MANNs with appending-based write and value or key-value memory
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory import *
from models import SimpleNTM

class SimpleNTMAppending(SimpleNTM):
    """SimpleNTM with

        -appending-based write
    """
    def __init__(
        self,
        # Basics
        encoder,
        encoder_output_size,
        model_output_size,
        batch_size,
        # Memory
        init_mem_size=10,
        mem_value_size=256,
        # Controller
        controller='lstm',
        controller_hidden_units=None,
        controller_output_size=128,
        # R/W head
        num_read_heads=1,
        num_write_heads=1):
        super(SimpleNTMAppending, self).__init__(
            # Basics
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=AppendingMemory,
            mem_size=init_mem_size,
            mem_value_size=mem_value_size,
            mem_extra_args=None,
            # Controller
            controller=controller,
            controller_hidden_units=controller_hidden_units,
            controller_output_size=controller_output_size,
            # R/W head
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            read_length=[mem_value_size],
            write_length=[mem_value_size]
        )
    
    def _write(self, x, prev_whead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.controller_output_size)
        :param prev_whead_state: list of None, length == len(self.write_heads)
        """
        ws = []
        for whead in self.write_heads:
            o = whead(x)
            self.memory.write_least_used(o)
            ws.append(None)
        return ws


class DND(nn.Module):
    """DND:

        -key-value based memory
        -simplified key-based reading
        -appending-based writing
        https://github.com/qihongl/dnd-lstm
        https://arxiv.org/pdf/1703.01988.pdf
        https://arxiv.org/pdf/1805.09692.pdf
    """
    def __init__(self):
        pass
    def forward(self):
        pass


class MRA(DND):
    """MRA:

        -similar to DND but with some changes on memory content
    """
    def __init__(self):
        pass
    def forward(self):
        pass
