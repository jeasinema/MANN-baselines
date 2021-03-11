import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mannbaselines.memory import *
from mannbaselines.utils import log

from mannbaselines.models.dnc import SimpleNTM, NTM
from mannbaselines.models.mra import MRA
from mannbaselines.models.mannmeta import MANNMeta
from mannbaselines.models.dnd import DND
from mannbaselines.models.raven import PositionalEncoding


class Encoder_conv(nn.Module):
    def __init__(self, args):
        super(Encoder_conv, self).__init__()
        log.info('Building convolutional encoder...')
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        # Fully-connected layers
        log.info('FC layers...')
        self.fc1 = nn.Linear(4*4*32, 256)
        self.fc2 = nn.Linear(256, 128)
        # Nonlinearities
        self.relu = nn.ReLU()
        # Initialize parameters
        for name, param in self.named_parameters():
            # Initialize all biases to 0
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            # Initialize all pre-ReLU weights using Kaiming normal distribution
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
    def forward(self, x):
        # Convolutional layers
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))
        # Flatten output of conv. net
        conv3_out_flat = torch.flatten(conv3_out, 1)
        # Fully-connected layers
        fc1_out = self.relu(self.fc1(conv3_out_flat))
        fc2_out = self.relu(self.fc2(fc1_out))
        # Output
        z = fc2_out
        return z

class Encoder_mlp(nn.Module):
    def __init__(self, args):
        super(Encoder_mlp, self).__init__()
        log.info('Building MLP encoder...')
        # Fully-connected layers
        log.info('FC layers...')
        self.fc1 = nn.Linear(32*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        # Nonlinearities
        self.relu = nn.ReLU()
        # Initialize parameters
        for name, param in self.named_parameters():
            # Initialize all biases to 0
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            # Initialize all pre-ReLU weights using Kaiming normal distribution
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
    def forward(self, x):
        # Flatten image
        x_flat = torch.flatten(x, 1)
        # Fully-connected layers
        fc1_out = self.relu(self.fc1(x_flat))
        fc2_out = self.relu(self.fc2(fc1_out))
        fc3_out = self.relu(self.fc3(fc2_out))
        # Output
        z = fc3_out
        return z

class Encoder_rand(nn.Module):
    def __init__(self, args):
        super(Encoder_rand, self).__init__()
        log.info('Building random encoder...')
        # Random projection
        self.fc1 = nn.Linear(32*32, 128)
        # Nonlinearities
        self.relu = nn.ReLU()
        # Initialize parameters
        for name, param in self.named_parameters():
            # Initialize all biases to 0
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            # Initialize all pre-ReLU weights using Kaiming normal distribution
            elif 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
    def forward(self, x):
        # Flatten image
        x_flat = torch.flatten(x, 1)
        # Fully-connected layers
        fc1_out = self.relu(self.fc1(x_flat)).detach()
        # Output
        z = fc1_out.detach()
        return z

class ESBN(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        self.memory.reset()
        pas

class ESBNNTM(nn.Module):
    def __init__(self, task_gen, args):
        super(ESBNNTM, self).__init__()
        self.output_dim = task_gen.y_dim
        # Encoder
        log.info('Building encoder...')
        if args.encoder == 'conv':
            self.encoder = Encoder_conv(args)
        elif args.encoder == 'mlp':
            self.encoder = Encoder_mlp(args)
        elif args.encoder == 'rand':
            self.encoder = Encoder_rand(args)
        # self.mann = NTM(
        #     self.encoder,
        #     128,
        #     self.output_dim,
        #     args.batch_size,
        #     # Memory
        #     mem_size=20,
        #     mem_value_size=512,
        #     # Controller
        #     controller='mlp',
        #     controller_hidden_units=None,
        #     controller_output_size=128,
        #     # R/W head
        #     num_read_heads=2,
        #     num_write_heads=2,
        #     head_beta_g_s_gamma=(1,1,3,1))
        self.mann = SimpleNTM(
            self.encoder,
            128,
            self.output_dim,
            args.batch_size,
            # Memory
            mem_size=10,
            mem_value_size=512,
            # Controller
            controller='lstm',
            controller_hidden_units=None,
            controller_output_size=128,
            # R/W head
            num_read_heads=1,
            num_write_heads=1)
        # self.mann = MRA(
        #     self.encoder,
        #     128,
        #     self.output_dim,
        #     args.batch_size,
        #     # Memory
        #     mem_size=20,
        #     mem_extra_args={'key_size': 256},
        #     # Controller
        #     controller='mlp',
        #     controller_hidden_units=None,
        #     controller_output_size=256,
        #     # R/W head
        #     num_read_heads=1,
        #     num_write_heads=1,
        #     k_nn=4,
        #     jumpy_bp=True)
        # self.mann = MANNMeta(
        #     self.encoder,
        #     128,
        #     self.output_dim,
        #     args.batch_size,
        #     # Memory
        #     mem_size=20,
        #     mem_value_size=512,
        #     mem_extra_args=None,
        #     # Controller
        #     controller='mlp',
        #     controller_hidden_units=None,
        #     controller_output_size=512,
        #     # R/W head
        #     num_read_heads=1,
        #     num_write_heads=1,
        #     # LRUA
        #     gamma=0.9)
        # self.mann = DND(
        #     self.encoder,
        #     128,
        #     self.output_dim,
        #     args.batch_size,
        #     # Memory
        #     mem_size=10,
        #     mem_extra_args={'key_size': 256},
        #     # Controller
        #     controller='lstm',
        #     controller_hidden_units=None,
        #     controller_output_size=128,
        #     # R/W head
        #     num_read_heads=1,
        #     num_write_heads=1,
        #     k_nn=1)

    def forward(self, x, device):
        # x: (B, T, 32, 32)
        B = x.size(0)
        T = x.size(1)
        x = x.view(B, T, 1, 32, 32).transpose(0, 1)
        y_pred_linear = self.mann(x)[0][-1]
        y_pred = y_pred_linear.argmax(1)
        return y_pred_linear, y_pred

class ESBNLSTM(nn.Module):
    def __init__(self, task_gen, args):
        super(ESBNLSTM, self).__init__()
        self.output_dim = task_gen.y_dim
        # Encoder
        log.info('Building encoder...')
        if args.encoder == 'conv':
            self.encoder = Encoder_conv(args)
        elif args.encoder == 'mlp':
            self.encoder = Encoder_mlp(args)
        elif args.encoder == 'rand':
            self.encoder = Encoder_rand(args)
        self.lstm = nn.LSTM(128, 512, batch_first=True)
        self.output_mlp = nn.Linear(512, self.output_dim)

    def forward(self, x, device):
        # x: (B, T, 32, 32)
        B = x.size(0)
        T = x.size(1)
        features = self.encoder(x.view(-1, 1, 32, 32)).reshape(B, T, -1)
        hidden = torch.zeros(1, B, 512).to(x)
        cell_state = torch.zeros(1, B, 512).to(x)
        final_features, _ = self.lstm(features, (hidden, cell_state))
        y_pred_linear = self.output_mlp(final_features[:, -1])
        y_pred = y_pred_linear.argmax(1)
        return y_pred_linear, y_pred

class ESBNTrans(nn.Module):
    def __init__(self, task_gen, args):
        super(ESBNTrans, self).__init__()
        self.output_dim = task_gen.y_dim
        # Encoder
        log.info('Building encoder...')
        if args.encoder == 'conv':
            self.encoder = Encoder_conv(args)
        elif args.encoder == 'mlp':
            self.encoder = Encoder_mlp(args)
        elif args.encoder == 'rand':
            self.encoder = Encoder_rand(args)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 8, dim_feedforward=512),
            4,
            nn.LayerNorm(128),
        )
        self.pos_emb = PositionalEncoding(128)
        self.output_mlp = nn.Linear(128, self.output_dim)

    def forward(self, x, device):
        # x: (B, T, 32, 32)
        B = x.size(0)
        T = x.size(1)
        features = self.encoder(x.view(-1, 1, 32, 32)).reshape(B, T, -1)
        features = self.pos_emb(features)
        # (B, T, ...) -> (T, B, ...)
        features = features.transpose(0, 1)
        final_features = self.transformer(features).mean(0)
        y_pred_linear = self.output_mlp(final_features)
        y_pred = y_pred_linear.argmax(1)
        return y_pred_linear, y_pred
