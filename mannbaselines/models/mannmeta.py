import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import haste_pytorch as haste

from mannbaselines.memory import *
from mannbaselines.models import SimpleNTM


class MANNMeta(SimpleNTM):
    """SimpleNTM with:

        -LURA-based write

    http://proceedings.mlr.press/v48/santoro16.pdf
    https://github.com/Leputa/MANN-meta-learning
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
        mem_extra_args=None,
        # Controller
        controller='lstm',
        controller_hidden_units=None,
        controller_output_size=128,
        # R/W head
        num_read_heads=1,
        num_write_heads=1,
        # LRUA
        gamma=0.9):
        super(MANNMeta, self).__init__(
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=ValueMemory,
            mem_size=mem_size,
            mem_value_size=mem_value_size,
            # Controller
            controller=controller,
            controller_hidden_units=controller_hidden_units,
            controller_output_size=controller_output_size,
            # R/W head
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            read_length=[mem_value_size],
            write_length=[mem_value_size, 1])
        self.gamma = gamma
        self.register_buffer('w_u', torch.FloatTensor(self.batch_size, self.memory.mem_size).fill_(0))

    def _write(self, x, prev_whead_state, prev_rhead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.controller_output_size)
        :param prev_whead_state: list of (B, self.memory.mem_size), length == len(self.write_heads)
        :param prev_rhead_state: list of (B, self.memory.mem_size), length == len(self.rrite_heads)
        """
        B = x.size(0)
        ws = []
        w_lus = []
        inds = torch.topk(self.w_u[:B], len(self.write_heads), -1, largest=False)[1].T
        w_lus = [torch.zeros(B, self.memory.mem_size).to(x).scatter_(1, ind.unsqueeze(-1), 1) for ind in inds]
        for whead, prev_rstate, w_lu in zip(self.write_heads, prev_rhead_state, w_lus):
            o = whead(x)
            v, alpha = split_cols(o, self.write_length)
            w = torch.sigmoid(alpha) * prev_rstate + (1 - torch.sigmoid(alpha)) * w_lu
            self.memory.clear(w_lu)
            self.memory.write(w, v)
            ws.append(w)
        return ws

    def forward_step(self, x, prev_latent=None):
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
        :param prev_latent: same as current latent state
        """
        B = x.size(0)
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
        whead_state = self._write(controller_output, prev_rwhead_state[len(self.read_heads):],
            prev_rwhead_state[:len(self.read_heads)])

        output = self.output_layer(torch.cat([controller_output] + reads, dim=-1))
        current_state = (controller_state, reads, rhead_state+whead_state)

        self.w_u[:B] = self.gamma * self.w_u[:B] + sum(whead_state) + sum(rhead_state)

        return output, current_state
