import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from .dnc import SimpleNTM, NTM
from .mra import MRA
from .mannmeta import MANNMeta
from .dnd import DND

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

    def train_(self, image, target, meta_target, meta_structure, embedding, indicator):
        self.optimizer.zero_grad()
        output = self(image, embedding, indicator)
        loss = self.compute_loss(output, target, meta_target, meta_structure)
        loss.backward()
        self.optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, image, target, meta_target, meta_structure, embedding, indicator):
        with torch.no_grad():
            output = self(image, embedding, indicator)
        loss = self.compute_loss(output, target, meta_target, meta_structure)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return loss.item(), accuracy

    def test_(self, image, target, meta_target, meta_structure, embedding, indicator):
        with torch.no_grad():
            output = self(image, embedding, indicator)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100.0 / target.size()[0]
        return accuracy

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x

class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8+9+21)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RAVENResnet18_MLP(RAVENBasicModel):
    def __init__(self, args):
        super(RAVENResnet18_MLP, self).__init__(args)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = identity()
        self.mlp = mlp_module()
        # self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred, meta_target_pred, meta_struct_pred = output[0], output[1], output[2]

        target_loss = F.cross_entropy(pred, target)
        meta_target_pred = torch.chunk(meta_target_pred, chunks=9, dim=1)
        meta_target = torch.chunk(meta_target, chunks=9, dim=1)
        meta_target_loss = 0.
        for idx in range(0, 9):
            meta_target_loss += F.binary_cross_entropy(torch.sigmoid(meta_target_pred[idx]), meta_target[idx])

        meta_struct_pred = torch.chunk(meta_struct_pred, chunks=21, dim=1)
        meta_structure = torch.chunk(meta_structure, chunks=21, dim=1)
        meta_struct_loss = 0.
        for idx in range(0, 21):
            meta_struct_loss += F.binary_cross_entropy(torch.sigmoid(meta_struct_pred[idx]), meta_structure[idx])
        loss = target_loss + self.meta_alpha*meta_struct_loss/21. + self.meta_beta*meta_target_loss/9.
        return loss

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
            nn.TransformerEncoderLayer(512, 8),
            4,
            nn.LayerNorm(512),
        )
        self.mlp = mlp_module()
        # self.fc_tree_net = FCTreeNet(in_dim=300, img_dim=512)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred, meta_target_pred, meta_struct_pred = output[0], output[1], output[2]

        target_loss = F.cross_entropy(pred, target)
        meta_target_pred = torch.chunk(meta_target_pred, chunks=9, dim=1)
        meta_target = torch.chunk(meta_target, chunks=9, dim=1)
        meta_target_loss = 0.
        for idx in range(0, 9):
            meta_target_loss += F.binary_cross_entropy(torch.sigmoid(meta_target_pred[idx]), meta_target[idx])

        meta_struct_pred = torch.chunk(meta_struct_pred, chunks=21, dim=1)
        meta_structure = torch.chunk(meta_structure, chunks=21, dim=1)
        meta_struct_loss = 0.
        for idx in range(0, 21):
            meta_struct_loss += F.binary_cross_entropy(torch.sigmoid(meta_struct_pred[idx]), meta_structure[idx])
        loss = target_loss + self.meta_alpha*meta_struct_loss/21. + self.meta_beta*meta_target_loss/9.
        return loss

    def forward(self, x, embedding, indicator):
        B = x.size(0)
        features = self.resnet18(x.view(-1, 1, 224, 224)).reshape(B, 16, -1)
        # TODO
        final_features = self.transformer(features).max(-2)[0]
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
        #     self.resnet18,
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
        #     num_write_heads=1,
        #     head_beta_g_s_gamma=(1,1,3,1))
        # self.mann = SimpleNTM(
        #     self.resnet18,
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
            self.resnet18,
            512,
            8+9+21,
            args.batch_size,
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
            jumpy_bp=True)
        # self.mann = MANNMeta(
        #     self.resnet18,
        #     512,
        #     8+9+21,
        #     args.batch_size
        #     # Memory
        #     mem_size=10,
        #     mem_value_size=256,
        #     mem_extra_args=None,
        #     # Controller
        #     controller='lstm',
        #     controller_hidden_units=None,
        #     controller_output_size=128,
        #     # R/W head
        #     num_read_heads=1,
        #     num_write_heads=1,
        #     # LRUA
        #     gamma=0.9)
        # self.mann = DND(
        #     self.resnet18,
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
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_alpha = args.meta_alpha
        self.meta_beta = args.meta_beta

    def compute_loss(self, output, target, meta_target, meta_structure):
        pred, meta_target_pred, meta_struct_pred = output[0], output[1], output[2]

        target_loss = F.cross_entropy(pred, target)
        meta_target_pred = torch.chunk(meta_target_pred, chunks=9, dim=1)
        meta_target = torch.chunk(meta_target, chunks=9, dim=1)
        meta_target_loss = 0.
        for idx in range(0, 9):
            meta_target_loss += F.binary_cross_entropy(torch.sigmoid(meta_target_pred[idx]), meta_target[idx])

        meta_struct_pred = torch.chunk(meta_struct_pred, chunks=21, dim=1)
        meta_structure = torch.chunk(meta_structure, chunks=21, dim=1)
        meta_struct_loss = 0.
        for idx in range(0, 21):
            meta_struct_loss += F.binary_cross_entropy(torch.sigmoid(meta_struct_pred[idx]), meta_structure[idx])
        loss = target_loss + self.meta_alpha*meta_struct_loss/21. + self.meta_beta*meta_target_loss/9.
        return loss

    def forward(self, x, embedding, indicator):
        # alpha = 1.0
        x = x.view(-1, 16, 1, 224, 224).transpose(0, 1)
        output = self.mann(x)[0][-1]
        pred = output[:,0:8]
        meta_target_pred = output[:,8:17]
        meta_struct_pred = output[:,17:38]
        return pred, meta_target_pred, meta_struct_pred
