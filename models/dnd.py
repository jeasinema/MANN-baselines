"""
A collection of MANNs with appending-based write and value or key-value memory
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import haste_pytorch as haste

from memory import *
from models import SimpleNTM

class SimpleNTMAppending(SimpleNTM):
    """SimpleNTM with

        -appending-based write
        -overwrite the least recently added entry
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
        num_write_heads=1):
        super(SimpleNTMAppending, self).__init__(
            # Basics
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=AppendingMemory,
            mem_size=mem_size,
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


class DND(SimpleNTM):
    """DND (SimpleNTMAppending w/ k-v mem):

        -key-value based memory
        -k-nn reading
        -appending-based writing
        -overwrite the least recently r/w entry

        https://github.com/qihongl/dnd-lstm
        https://arxiv.org/pdf/1703.01988.pdf
        https://arxiv.org/pdf/1805.09692.pdf
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
        mem_extra_args={'key_size': 256},
        # Controller
        controller='lstm',
        controller_hidden_units=None,
        controller_output_size=128,
        # R/W head
        num_read_heads=1,
        num_write_heads=1,
        k_nn=1):
        self.k_nn = k_nn
        super(DND, self).__init__(
            # Basics
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=DNDMemory,
            mem_size=mem_size,
            mem_value_size=mem_value_size,
            mem_extra_args=mem_extra_args,
            # Controller
            controller=controller,
            controller_hidden_units=controller_hidden_units,
            controller_output_size=controller_output_size,
            # R/W head
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            read_length=[mem_extra_args['key_size']],
            write_length=[mem_extra_args['key_size'], mem_value_size]
        )

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
            k = rhead(x)
            w = self._address(k, self.k_nn)
            r = self.memory.read(w)
            ws.append(w)
            reads.append(r)
        return reads, ws

    def _write(self, x, prev_whead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.controller_output_size)
        :param prev_whead_state: list of None, length == len(self.write_heads)
        """
        ws = []
        for whead in self.write_heads:
            o = whead(x)
            k, v = split_cols(o, self.write_length)
            self.memory.write_least_used(k, v)
            ws.append(None)
        return ws


class DNDCustomizedValue(SimpleNTM):
    """DND with:

        -value can be customized, does not need to be produced by write head.

        https://github.com/qihongl/dnd-lstm
        https://arxiv.org/pdf/1703.01988.pdf
        https://arxiv.org/pdf/1805.09692.pdf
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
        mem_extra_args={'key_size': 256},
        # Controller
        controller='lstm',
        controller_hidden_units=None,
        controller_output_size=128,
        # R/W head
        num_read_heads=1,
        num_write_heads=1,
        k_nn=1,
        jumpy_bp=True):
        self.k_nn = k_nn
        self.jumpy_bp = jumpy_bp
        super(DNDCustomizedValue, self).__init__(
            # Basics
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=DNDMemory,
            mem_size=mem_size,
            mem_value_size=mem_value_size,
            mem_extra_args=mem_extra_args,
            # Controller
            controller=controller,
            controller_hidden_units=controller_hidden_units,
            controller_output_size=controller_output_size,
            # R/W head
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            read_length=[mem_extra_args['key_size']],
            write_length=[mem_extra_args['key_size']]
        )

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
            k = rhead(x)
            w = self._address(k, self.k_nn)
            r = self.memory.read(w)
            ws.append(w)
            reads.append(r)
        return reads, ws

    def _write(self, x, v, prev_whead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.controller_output_size)
        :param v: shape (B, self.memory.mem_value_size)
        :param prev_whead_state: list of None, length == len(self.write_heads)
        """
        ws = []
        for whead in self.write_heads:
            if self.jumpy_bp:
                k = whead(x.detach())
            else:
                k = whead(x)
            self.memory.write_least_used(k, v)
            ws.append(None)
        return ws

    def forward_step(self, x, v, prev_latent=None):
        """
        :output ntm output: shape (B, self.model_output_size)
        :output current latent state:
            tuple(
                shape (self.controller_num_layer, B, self.controller_output_size) -- LSTM
                      (B, self.controller_output_size) -- MLP
                    -- controller latent state, required by controller
                shape list of (B, self.memory.value_size), length == len(self.read_heads)
                    -- init read, required by controller
                shape list of (B, self.memory.mem_size), length == len(self.read_heads+self.write_heads)
                    -- init head rw/ weight, required by read/write head (interpolation)
            )

        :param x: shape (B, ...)
        :param v: shape (B, self.memory.mem_value_size)
        :param prev_latent: same as current latent state
        """
        if not prev_latent:
            prev_latent = self.init_state(x.size(0))
        prev_controller_state, prev_reads, prev_rwhead_state = prev_latent

        x_emb = self.encoder(x)
        controller_input = torch.cat([x_emb] + prev_reads, dim=-1)
        if self.controller_type == 'lstm':
            controller_output, controller_state = self.controller(
                controller_input.unsqueeze(0),
                prev_controller_state
            )
            controller_output = controller_output[0]
        else:
            controller_output = self.controller(controller_input)
            controller_state = None

        reads, rhead_state = self._read(controller_output, prev_rwhead_state[:len(self.read_heads)])
        whead_state = self._write(controller_output, v, prev_rwhead_state[len(self.read_heads):])

        output = self.output_layer(torch.cat([controller_output] + reads, dim=-1))
        current_state = (controller_state, reads, rhead_state+whead_state)

        return output, current_state

    def forward(self, x, v, init_latent=None):
        """
        :output ntm output: shape (T, B, self.model_output_size)
        :output current latent state:
            tuple(
                shape (self.controller_num_layer, B, self.controller_output_size) -- LSTM
                      (B, self.controller_output_size) -- MLP
                shape list of (B, self.memory.value_size), length == len(self.read_heads)
                    -- init read, required by controller
                shape list of (B, self.memory.mem_size), length == len(self.read_heads+self.write_heads)
                    -- init head rw/ weight, required by read/write head (interpolation)
            )

        :param  x: shape (T, B, ...)
        :param  v: shape (T, B, self.memory.mem_value_size)
        :param init_latent: shape same as current latent state
        """
        T = x.size(0)
        outputs = []
        prev_latent = init_latent
        for i in range(T):
            o, prev_latent = self.forward_step(x[i], v[i], prev_latent)
            outputs.append(o)

        return torch.stack(outputs), prev_latent


class DNDRL(SimpleNTM):
    """
    Complete DND (https://arxiv.org/pdf/1805.09692.pdf)
        where the LSTM hidden state is saved and read from the memory.
    """
    def __init__(self):
        pass
    def forward(self):
        pass