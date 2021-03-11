import torch
import torch.autograd as autograd
import math
import numpy as np
import torch.nn.functional as F

from mannbaselines.utils.torch import use_gpu


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - \
        0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def normal_kl_div(mu1, mu2, sigma1, sigma2):
    sub = torch.ones_like(mu1)*0.5
    return (torch.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*(sigma2**2))
            - sub).sum(-1).mean()

def multinormal_kl_div(mu1, mu2, cov1, cov2):
    d = mu1.shape[-1]
    a = torch.log(torch.det(cov2)/torch.det(cov1))
    b = torch.diagonal(torch.matmul(torch.inverse(cov2), cov1), dim1=-2, dim2=-1).sum(-1)
    c = torch.bmm(torch.bmm((mu2-mu1).unsqueeze(1), torch.inverse(cov2)),
                  (mu2-mu1).unsqueeze(-1))
    return (1/2 * (a + b + c - d)).mean()

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, LAMBDA=10):
    # print real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_gpu else alpha

    differences = fake_data - real_data
    interpolates = real_data + alpha * differences
    # interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.requires_grad_()

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_gpu else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# refer to: https://github.com/nosyndicate/pytorchrl/blob/c4fb69ffebaa7f56b4210388f9eea7d42ca853e4/pytorchrl/misc/tensor_utils.py#L5
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# refer to: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps).to(device)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=0.8):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y
