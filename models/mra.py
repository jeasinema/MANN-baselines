import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import haste_pytorch as haste

from memory import *
from models import SimpleNTM


class MRA(SimpleNTM):
    """MRA:

        -k-v-p memory, while only the v is read from the memory
        -controller takes current head and input emb
        -read head takes previous controller output and current input emb and knn
        -appending-based write, value 
        -jumpy bp
        https://arxiv.org/pdf/1910.13406.pdf
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
        super(MRA, self).__init__(
            # Basics
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=MRAMemory,
            mem_size=mem_size,
            mem_value_size=controller_output_size+encoder_output_size,
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
        # Overwrite controller, w/rhead and output layer
        del self.controller
        del self.read_heads
        del self.write_heads
        del self.output_layer
        del self.init_read
        controller_hidden_units = controller_hidden_units if controller_hidden_units else []
        if controller == 'mlp':
            self.controller = []
            last_dim = self.encoder_output_size + num_read_heads * self.controller_output_size
            for nh in controller_hidden_units:
                self.controller.append(nn.Linear(last_dim, nh))
                self.controller.append(nn.ReLU())
                last_dim = nh
            self.controller.append(nn.Linear(last_dim, controller_output_size))
            self.controller = nn.Sequential(*self.controller)
            self.register_parameter('init_mlp_o', nn.Parameter(torch.FloatTensor(1, self.controller_output_size).fill_(0)))
        elif controller == 'lstm':
            assert self.controller_num_layer == 1
            if self.controller_num_layer > 1:
                assert set(controller_hidden_units) == 1
                assert controller_hidden_units[0] == controller_output_size
            self.controller = haste.LayerNormLSTM(
                self.encoder_output_size + num_read_heads * self.controller_output_size,
                self.controller_output_size,
                )
        self.read_heads = nn.ModuleList([
            nn.Linear(self.encoder_output_size+self.controller_output_size, sum(self.read_length))
            for _ in range(num_read_heads)
        ])
        self.write_heads = nn.ModuleList([
            nn.Linear(self.encoder_output_size+self.controller_output_size, sum(self.write_length))
            for _ in range(num_write_heads)
        ])
        # TODO: zero vs. rand init
        # If random, then should be different across different heads
        self.register_parameter('init_read', nn.Parameter(torch.FloatTensor(1, self.controller_output_size).fill_(0)))
        self.output_layer = nn.Linear(
            self.controller_output_size + self.controller_output_size * num_read_heads,
            self.model_output_size
        )
        # Initialize again for the overwritten modules
        self.reset(reset_memory=False)

    def _read(self, x, prev_rhead_state):
        """
        :output read value: shape list of (B, self.memory.value_size), length == len(self.read_heads)
        :output r head weight: shape list of (B, self.memory.mem_size), length == len(self.read_heads)

        :param x: shape (B, self.encoder_output_size+self.controller_output_size)
        :param prev_rhead_state: list of (B, self.memory.mem_size), length == len(self.read_heads)
        """
        reads = []
        ws = []
        for rhead in self.read_heads:
            k = rhead(x)
            w = self._address(k, self.k_nn)
            # Only v is read
            r = self.memory.read(w)[:, -self.controller_output_size:]
            ws.append(w)
            reads.append(r)
        return reads, ws

    def _write(self, x, prev_whead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.encoder_output_size+self.controller_output_size)
        :param prev_whead_state: list of None, length == len(self.write_heads)
        """
        ws = []
        for whead in self.write_heads:
            if self.jumpy_bp:
                k = whead(x.detach())
            else:
                k = whead(x)
            self.memory.write_least_used(k, x)
            ws.append(None)
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
        if not prev_latent:
            prev_latent = self.init_state(x.size(0))
        prev_controller_state, prev_reads, prev_rwhead_state = prev_latent

        x_emb = self.encoder(x)
        if self.controller_type == 'lstm':
            rhead_input = torch.cat([x_emb, prev_controller_state[0].transpose(0, 1)[:, 0]], dim=-1)
        else:
            rhead_input = torch.cat([x_emb, prev_controller_state], dim=-1)
        reads, rhead_state = self._read(rhead_input, prev_rwhead_state[:len(self.read_heads)])

        controller_input = torch.cat([x_emb] + reads, dim=-1)
        if self.controller_type == 'lstm':
            controller_output, controller_state = self.controller(
                controller_input.unsqueeze(0),
                prev_controller_state
            )
            controller_output = controller_output[0]
        else:
            controller_output = self.controller(controller_input)
            controller_state = controller_output.detach()

        whead_input = torch.cat([x_emb, controller_output], dim=-1)
        whead_state = self._write(whead_input, prev_rwhead_state[len(self.read_heads):])

        output = self.output_layer(torch.cat([controller_output] + reads, dim=-1))
        current_state = (controller_state, reads, rhead_state+whead_state)

        return output, current_state

    def forward(self, x, init_latent=None):
        """
        :output ntm output: shape (T, B, self.model_output_size)
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

        :param  x: shape (T, B, ...)
        :param init_latent: shape same as current latent state
        """
        self.memory.reset()
        T = x.size(0)
        outputs = []
        prev_latent = init_latent
        for i in range(T):
            o, prev_latent = self.forward_step(x[i], prev_latent)
            outputs.append(o)

        return torch.stack(outputs), prev_latent
