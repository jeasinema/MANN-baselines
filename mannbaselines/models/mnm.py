import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mannbaselines.memory import *

class MNM(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        self.memory.reset()
        pass
