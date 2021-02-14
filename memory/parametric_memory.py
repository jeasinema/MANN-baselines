import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class ParametricMemory(nn.Module):

    def __init__(self):
        super(ParametricMemory, self).__init__()