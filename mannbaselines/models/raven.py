import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from mannbaselines.models.dnc import SimpleNTM, NTM
from mannbaselines.models.mra import MRA
from mannbaselines.models.mannmeta import MANNMeta
from mannbaselines.models.dnd import DND

def apply_context_norm(z_seq, gamma, beta, dim=1):
    eps = 1e-8
    z_mu = z_seq.mean(dim)
    z_sigma = (z_seq.var(dim) + eps).sqrt()
    z_seq = (z_seq - z_mu.unsqueeze(dim)) / z_sigma.unsqueeze(dim)
    z_seq = (z_seq * gamma) + beta
    return z_seq

class RAVENBasicModel(nn.Module):
    def __init__(self, args):
        super(RAVENBasicModel, self).__init__()
        self.name = args.model

    def load_model(self, path, epoch):
        state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch))['state_dict']
        self.load_state_dict(state_dict)

    def save_model(self, path, epoch, acc, loss):
        torch.save({'state_dict': self.state_dict(), 'acc': acc, 'loss': loss}, path+'{}_epoch_{}.pth'.format(self.name, epoch))

    def compute_loss(self, output, target, meta_target, meta_structure):
        pass

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x

class mlp_module(nn.Module):
    def __init__(self, input_dim=512):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8+9+21)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # Input (B, T, ...)
        x = x + self.pe[:, :x.size(1), :]
        return x

class RAVENResnet18_MLP(RAVENBasicModel):
    def __init__(self, args):
        super(RAVENResnet18_MLP, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.mlp = mlp_module()
        # self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def forward(self, x, embedding, indicator):
        # alpha = 1.0
        features = self.resnet18(x.view(-1, 16, 224, 224))
        # features_tree = features.view(-1, 1, 512)
        # features_tree = self.fc_tree_net(features_tree, embedding, indicator)
        # final_features = features + alpha * features_tree
        final_features = features
        output = self.mlp(final_features)
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred


class RAVENTrans(RAVENBasicModel):
    def __init__(self, args):
        super(RAVENTrans, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8, dim_feedforward=512),
            1,
            nn.LayerNorm(512),
        )
        self.pos_emb = PositionalEncoding(512)
        self.mlp = mlp_module(512)
        # self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta
        # Context norm
        self.gamma = nn.Parameter(torch.ones(512))
        self.beta = nn.Parameter(torch.zeros(512))
        self.task_seg = [np.arange(16)]

    def forward(self, x, embedding, indicator):
        B = x.size(0)
        features = self.resnet18(x.view(-1, 1, 224, 224)).reshape(B, 16, -1)
        z_seq_all_seg = []
        for seg in range(len(self.task_seg)):
            z_seq_all_seg.append(apply_context_norm(features[:, self.task_seg[seg], :], self.gamma, self.beta, dim=1))
        features = torch.cat(z_seq_all_seg, dim=1)
        features = self.pos_emb(features)
        # (B, T, ...) -> (T, B, ...) as required by the transformer
        features = features.transpose(0, 1)
        final_features = self.transformer(features).mean(0)
        output = self.mlp(final_features)
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred


class RAVENNTM(RAVENBasicModel):
    def __init__(self, args):
        super(RAVENNTM, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        # self.mann = NTM(
        #     None,
        #     512,
        #     8+9+21,
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
        # self.mann = SimpleNTM(
        #     None,
        #     512,
        #     8+9+21,
        #     args.batch_size,
        #     # Memory
        #     mem_size=10,
        #     mem_value_size=512,
        #     # Controller
        #     controller='lstm',
        #     controller_hidden_units=None,
        #     controller_output_size=128,
        #     # R/W head
        #     num_read_heads=1,
        #     num_write_heads=1)
        self.mann = MRA(
            None,
            512,
            8+9+21,
            args.batch_size,
            # Memory
            mem_size=20,
            mem_extra_args={'key_size': 256},
            # Controller
            controller='lstm',
            controller_hidden_units=None,
            controller_output_size=256,
            # R/W head
            num_read_heads=1,
            num_write_heads=1,
            k_nn=4,
            jumpy_bp=True)
        # self.mann = MANNMeta(
        #     None,
        #     512,
        #     8+9+21,
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
        #     None,
        #     512,
        #     8+9+21,
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
        # self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta
        # Context norm
        self.gamma = nn.Parameter(torch.ones(512))
        self.beta = nn.Parameter(torch.zeros(512))
        self.task_seg = [np.arange(16)]

    def forward(self, x, embedding, indicator):
        # alpha = 1.0
        B = x.size(0)
        T = x.size(1)
        x = x.view(-1, 16, 1, 224, 224).view(-1, 1, 224, 224)
        features = self.resnet18(x).view(B, T, -1)
        z_seq_all_seg = []
        for seg in range(len(self.task_seg)):
            z_seq_all_seg.append(apply_context_norm(features[:, self.task_seg[seg], :], self.gamma, self.beta, dim=1))
        features = torch.cat(z_seq_all_seg, dim=1).transpose(0, 1)
        output = self.mann(features)[0][-1]
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred

############################
class RAVENL3(RAVENBasicModel):
    def __init__(self, args):
        super(RAVENL3, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        # 3x3
        self.pos_num_rule_pred = nn.Linear(512, 17+1)
        self.type_rule_pred = nn.Linear(512, 7+1)
        self.size_rule_pred = nn.Linear(512, 7+1)
        self.color_rule_pred = nn.Linear(512, 9+1)
        # 2x2
        self.pos_num_rule_pred = nn.Linear(512, 13+1)
        self.type_rule_pred = nn.Linear(512, 7+1)
        self.size_rule_pred = nn.Linear(512, 7+1)
        self.color_rule_pred = nn.Linear(512, 9+1)
        # others (single_center)
        self.pos_num_rule_pred = nn.Linear(512, 0+1)
        self.type_rule_pred = nn.Linear(512, 7+1)
        self.size_rule_pred = nn.Linear(512, 7+1)
        self.color_rule_pred = nn.Linear(512, 9+1)

        self.mlp = mlp_module()

    def forward(self, x, target, rule_target):
        B = x.size(0)
        features = self.resnet18(x.view(-1, 16, 224, 224))
        final_features = features
        output = self.mlp(final_features)
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred

def compute_loss_l3(args, output, target, rule_target):
    pass

def pred_l3(args, x):
    pass

class RAVENEBCL(RAVENBasicModel):
    def __init__(self, args):
        super(RAVENEBCL, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.mlp = mlp_module()
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def forward(self, x, target, rule_target):
        features = self.resnet18(x.view(-1, 16, 224, 224))
        final_features = features
        output = self.mlp(final_features)
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred

def compute_loss_ebcl(args, output, target, rule_target):
    pass

def pred_ebcl(args, x):
    pass
