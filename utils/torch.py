import torch
import numpy as np

use_gpu = torch.cuda.is_available()
DoubleTensor = torch.cuda.DoubleTensor if use_gpu else torch.DoubleTensor
FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor

LOG_EPS=1e-16

def to_one_hot(x, dim):
    # Input:
    #   x: [0,2,3,1...] discrete numbers, 1 dim
    #   dim: size of the one hot tensor
    tmp = torch.eye(dim).to(x.device)
    return tmp[x.squeeze()]

def ones(*shape):
    return torch.ones(*shape).cuda() if use_gpu else torch.ones(*shape)


def zeros(*shape, **kwargs):
    return torch.zeros(*shape).cuda() if use_gpu else torch.zeros(*shape)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params
    # return torch.nn.utils.parameters_to_vector(model.parameters())


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    # torch.nn.utils.vector_to_parameters(flat_params, model.parameters())


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1).detach())
        else:
            grads.append(param.grad.view(-1).detach())

    flat_grad = torch.cat(grads)
    return flat_grad


def set_lr(optimizer, lr):
    s = optimizer.state_dict()
    s['param_groups'][0]['lr'] = lr
    optimizer.load_state_dict(s)


def split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


def squash(x, outer_dims):
    assert x.dim() > outer_dims
    return x.reshape([-1] + list(x.shape[outer_dims:])), list(x.shape)

def unsquash(x, orig_shape):
    return x.reshape(orig_shape)