"""
A collection of MANNs with value-based memory and addressing-based r/w.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import haste_pytorch as haste

from mannbaselines.memory import *


class SimpleNTM(nn.Module):
    """
    NTM with simplified r/w (simplifed content-based only addressing)
    https://github.com/loudinthecloud/pytorch-ntm
    """
    def __init__(
        self,
        # Basics
        encoder,
        encoder_output_size,
        model_output_size,
        batch_size,
        # Memory
        mem_class=ValueMemory,
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
        read_length=None,
        write_length=None):
        super(SimpleNTM, self).__init__()
        self.encoder = encoder if encoder else nn.Identity()
        # output size of the encoder
        self.encoder_output_size = encoder_output_size
        self.model_output_size = model_output_size
        self.batch_size = batch_size

        mem_extra_args = mem_extra_args if mem_extra_args else {}
        self.memory = mem_class(mem_size, mem_value_size, batch_size, **mem_extra_args)

        self.controller_type = controller
        controller_hidden_units = controller_hidden_units if controller_hidden_units else []
        self.controller_output_size = controller_output_size
        self.controller_num_layer = len(controller_hidden_units) + 1
        if controller == 'mlp':
            self.controller = []
            last_dim = self.encoder_output_size + num_read_heads * mem_value_size
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
                self.encoder_output_size + num_read_heads * mem_value_size,
                self.controller_output_size,
                )
            # TODO: zero vs. rand init
            self.register_parameter('init_lstm_h', nn.Parameter(torch.FloatTensor(self.controller_num_layer, 1, self.controller_output_size).fill_(0)))
            self.register_parameter('init_lstm_c', nn.Parameter(torch.FloatTensor(self.controller_num_layer, 1, self.controller_output_size).fill_(0)))
        else:
            raise NotImplementedError

        self.read_length = read_length if read_length else [mem_value_size]
        self.write_length = write_length if write_length else [mem_value_size, mem_value_size]
        self.read_heads = nn.ModuleList([
            nn.Linear(self.controller_output_size, sum(self.read_length))
            for _ in range(num_read_heads)
        ])
        self.write_heads = nn.ModuleList([
            nn.Linear(self.controller_output_size, sum(self.write_length))
            for _ in range(num_write_heads)
        ])
        # TODO: zero vs. rand init
        # If random, then should be different across different heads
        self.register_parameter('init_read', nn.Parameter(torch.FloatTensor(1, mem_value_size).fill_(0)))

        self.output_layer = nn.Linear(
            self.controller_output_size + mem_value_size * num_read_heads,
            self.model_output_size
        )

        self.reset(reset_memory=True)

    def reset(self, reset_memory=True):
        if reset_memory:
            self.memory.reset()

        for name, p in self.controller.named_parameters():
            if self.controller_type == 'mlp':
                if 'weight' in name:
                    nn.init.xavier_uniform_(p, gain=1.4)
                elif 'bias' in name:
                    nn.init.normal_(p, std=0.01)
                else:
                    raise NotImplementedError
            elif self.controller_type == 'lstm':
                if p.dim() == 1:
                    nn.init.constant_(p, 0)
                else:
                    stdev = 5 / (np.sqrt(self.encoder_output_size +  self.controller_output_size))
                    nn.init.uniform_(p, -stdev, stdev)
        for head in self.read_heads:
            nn.init.xavier_uniform_(head.weight, gain=1.4)
            nn.init.normal_(head.bias, std=0.01)
        for head in self.write_heads:
            nn.init.xavier_uniform_(head.weight, gain=1.4)
            nn.init.normal_(head.bias, std=0.01)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=1.4)
        nn.init.normal_(self.output_layer.bias, std=0.01)

    def init_state(self, batch_size=-1):
        """
        :output controller latent state:
            shape (self.controller_num_layer, B, self.controller_output_size) -- LSTM
                  (B, self.controller_output_size) -- MLP
                required by controller
        :output init read values:
            shape list of (B, self.memory.value_size), length == len(self.read_heads)
                required by controller
        :output init r/w head weight:
            shape list of (B, self.memory.mem_size), length == len(self.read_heads+self.write_heads)
                required by read/write head (interpolation)
        """
        B = batch_size if batch_size else self.batch_size
        if self.controller_type == 'lstm':
            init_controller_state = (
                self.init_lstm_h.detach().repeat(1, B, 1),
                self.init_lstm_c.detach().repeat(1, B, 1),
            )
        else:
            init_controller_state = self.init_mlp_o.detach().repeat(B, 1)
        init_reads = [
            self.init_read.detach().repeat(B, 1)
            for _ in self.read_heads
        ]
        # TODO: zero vs. rand init
        init_rwhead_state = [
            torch.zeros(B, self.memory.shape[0]).to(init_reads[0])
            for _ in range(len(self.read_heads) + len(self.write_heads))
        ]
        return init_controller_state, init_reads, init_rwhead_state

    def _address(self, k, topk=-1):
        """NTM Addressing (according to section 3.3).
        :output weight: shape (B, self.memory.mem_size)

        :param k: shape (B, .) The key vector.
        """
        # Content focus
        sim = self.memory.similarity(k, topk=topk)
        w = F.softmax(sim, dim=1)

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
            k = rhead(x)
            w = self._address(k)
            r = self.memory.read(w)
            ws.append(w)
            reads.append(r)
        return reads, ws

    def _write(self, x, prev_whead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.controller_output_size)
        :param prev_whead_state: list of (B, self.memory.mem_size), length == len(self.write_heads)
        """
        ws = []
        for whead in self.write_heads:
            o = whead(x)
            k, v = split_cols(o, self.write_length)
            w = self._address(k)
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
        whead_state = self._write(controller_output, prev_rwhead_state[len(self.read_heads):])

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


class NTM(SimpleNTM):
    """
    Complete NTM (content-based + localtion-based addressing)
        https://github.com/loudinthecloud/pytorch-ntm
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
        head_beta_g_s_gamma=(1,1,3,1)):
        super(NTM, self).__init__(
            # Basics
            encoder=encoder,
            encoder_output_size=encoder_output_size,
            model_output_size=model_output_size,
            batch_size=batch_size,
            # Memory
            mem_class=NTMMemory,
            mem_size=mem_size,
            mem_value_size=mem_value_size,
            # Controller
            controller=controller,
            controller_hidden_units=controller_hidden_units,
            controller_output_size=controller_output_size,
            # R/W head
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
            read_length=[mem_value_size, *head_beta_g_s_gamma],
            write_length=[mem_value_size, *head_beta_g_s_gamma, mem_value_size, mem_value_size]
        )

    def _convolve(self, w, s):
        """Circular convolution implementation."""
        assert s.size(0) == 3
        t = torch.cat([w[-1:], w, w[:1]])
        c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
        return c

    def _address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).
        :output weight: shape (B, self.memory.mem_size)

        :param k: shape (B, .) The key vector.
        :param β: shape (B, .) The key strength (focus).
        :param g: shape (B, .) Scalar interpolation gate (with previous weight).
        :param s: shape (B, .) Shift weight.
        :param γ: shape (B, .) Sharpen weight scalar.
        :param w_prev: shape (B, self.memory.mem_size) The weight produced in the previous time step.
        """
        # Handle Activations
        β = F.softplus(β)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        # Content focus
        sim = self.memory.similarity(k)
        wc = F.softmax(β * sim, dim=1)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = torch.zeros_like(wg)
        for b in range(wg.size(0)):
            result[b] = self._convolve(wg[b], s[b])
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
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
        for rhead, w_prev in zip(self.read_heads, prev_rhead_state):
            o = rhead(x)
            k, β, g, s, γ = split_cols(o, self.read_length)
            w = self._address(k, β, g, s, γ, w_prev)
            r = self.memory.read(w)
            ws.append(w)
            reads.append(r)
        return reads, ws

    def _write(self, x, prev_whead_state):
        """
        :output w head weight: shape list of (B, self.memory.mem_size), length == len(self.write_heads)

        :param x: shape (B, self.controller_output_size)
        :param prev_whead_state: list of (B, self.memory.mem_size), length == len(self.write_heads)
        """
        ws = []
        for whead, w_prev in zip(self.write_heads, prev_whead_state):
            o = whead(x)
            k, β, g, s, γ, e, a = split_cols(o, self.write_length)
            e = torch.sigmoid(e)
            w = self._address(k, β, g, s, γ, w_prev)
            self.memory.write(w, e, a)
            ws.append(w)
        return ws


class DNC(nn.Module):
    """DNC:

        NTM with learnable allocation.
        https://github.com/ixaxaar/pytorch-dnc
    """
    def __init__(self):
        pass
    def forward(self):
        self.memory.reset()
        pass
