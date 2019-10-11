import argparse
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

multiply_adds = 1

def non_sparsity(weight):
    return (1- torch.sum(torch.abs(weight.flatten()) == 0.).item()/ weight.numel())

def zero_ops(m, x, y):
    m.total_add_ops += torch.Tensor([0])
    m.total_mul_ops += torch.Tensor([0])
    
    
def count_convNd(m, x, y):
    x = x[0]

    kernel_ops = m.weight.size()[2:].numel()  # Kw x Kh
    bias_ops = 1 if m.bias is not None else 0
    
    total_add_ops =  y.nelement() * (m.in_channels // m.groups * kernel_ops - 1 + bias_ops) * non_sparsity(m.weight)
    total_mul_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops) * non_sparsity(m.weight)

    m.total_add_ops += torch.Tensor([total_add_ops])
    m.total_mul_ops += torch.Tensor([total_mul_ops])



def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    # subtract, divide, gamma, beta

    m.total_add_ops += torch.Tensor([nelements]) * 2
    m.total_mul_ops += torch.Tensor([nelements]) * 2

def hswish_ops(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_add_ops += torch.Tensor([nelements])
    m.total_mul_ops += torch.Tensor([nelements]) * 4
    
def sigmoid_ops(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_add_ops += torch.Tensor([nelements])
    m.total_mul_ops += torch.Tensor([nelements]) * 2
    


def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size]))
    total_div = 1
    num_elements = y.numel()
    total_add_ops = total_add * num_elements
    total_mul_ops = total_div * num_elements
    
    m.total_add_ops += torch.Tensor([total_add_ops])
    m.total_mul_ops += torch.Tensor([total_mul_ops])

def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel)
    total_div = 1
    num_elements = y.numel()
    total_add_ops = total_add * num_elements
    total_mul_ops = total_div * num_elements
    
    m.total_add_ops += torch.Tensor([total_add_ops])
    m.total_mul_ops += torch.Tensor([total_mul_ops])


def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_add_ops = total_add * num_elements * non_sparsity(m.weight)
    total_mul_ops = total_mul * num_elements * non_sparsity(m.weight)

    m.total_add_ops += torch.Tensor([total_add_ops]) 
    m.total_mul_ops += torch.Tensor([total_mul_ops]) 