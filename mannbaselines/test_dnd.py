#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : test.py
# Creation Date : 17-02-2021
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch
import torch.nn as nn
from models import SimpleNTMAppending, DND, DNDCustomizedValue, MRA

if __name__ == '__main__':
    m = nn.Linear(8, 10)
    ntm = SimpleNTMAppending(m, 10, 2, 10).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('SimpleNTMAppending lstm done')

    ntm = SimpleNTMAppending(m, 10, 2, 10, num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('SimpleNTMAppending lstm multihead done')

    ntm = SimpleNTMAppending(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64)).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('SimpleNTMAppending mlp done')

    ntm = SimpleNTMAppending(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64), num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('SimpleNTMAppending mlp multihead done')

    m = nn.Linear(8, 10)
    ntm = DND(m, 10, 2, 10).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('DND lstm done')

    ntm = DND(m, 10, 2, 10, num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('DND lstm multihead done')

    ntm = DND(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64)).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('DND mlp done')

    ntm = DND(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64), num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('DND mlp multihead done')

    m = nn.Linear(8, 10)
    ntm = DNDCustomizedValue(m, 10, 2, 10).cuda()
    a = torch.ones(10, 8).cuda()
    v = torch.rand(10, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    v = torch.rand(8, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    v = torch.rand(16, 10, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    v = torch.rand(16, 8, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    print('DNDCustomizedValue lstm done')

    ntm = DNDCustomizedValue(m, 10, 2, 10, num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    v = torch.rand(10, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    v = torch.rand(8, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    v = torch.rand(16, 10, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    v = torch.rand(16, 8, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    print('DNDCustomizedValue lstm multihead done')

    ntm = DNDCustomizedValue(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64)).cuda()
    a = torch.ones(10, 8).cuda()
    v = torch.rand(10, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    v = torch.rand(8, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    v = torch.rand(16, 10, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    v = torch.rand(16, 8, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    print('DNDCustomizedValue mlp done')

    ntm = DNDCustomizedValue(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64), num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    v = torch.rand(10, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    v = torch.rand(8, 256).cuda()
    print(a.shape)
    print(ntm.forward_step(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    v = torch.rand(16, 10, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    v = torch.rand(16, 8, 256).cuda()
    print(a.shape)
    print(ntm.forward(a, v)[0].shape)
    print('===========')
    print('DNDCustomizedValue mlp multihead done')

    m = nn.Linear(8, 10)
    ntm = MRA(m, 10, 2, 10).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('MRA lstm done')

    ntm = MRA(m, 10, 2, 10, num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('MRA lstm multihead done')

    ntm = MRA(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64)).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('MRA mlp done')

    ntm = MRA(m, 10, 2, 10, controller='mlp', controller_hidden_units=(128, 64), num_read_heads=3, num_write_heads=3).cuda()
    a = torch.ones(10, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(8, 8).cuda()
    print(a.shape)
    print(ntm.forward_step(a)[0].shape)
    print('===========')
    a = torch.ones(16, 10, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    a = torch.ones(16, 8, 8).cuda()
    print(a.shape)
    print(ntm.forward(a)[0].shape)
    print('===========')
    print('MRA mlp multihead done')