import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .count_hooks import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: zero_ops,
    nn.BatchNorm2d: zero_ops,
    nn.BatchNorm3d: zero_ops,

    nn.ReLU6: hswish_ops,
    nn.Sigmoid: sigmoid_ops,
    
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
}


def count(model, inputs, custom_ops=None, verbose=True):
    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_add_ops") or hasattr(m, "total_params") or hasattr(m, "total_mul_ops"):
            logger.warning("Either .total_add_ops or .total_mul_ops or .total_params is already defined in %s."
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_add_ops', torch.zeros(1))
        m.register_buffer('total_mul_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        #for p in m.parameters():
        #    if p.ndimension() != 1:
        #        m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops: 
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("Count has not implemented counting method for ", m)
        else:
            if verbose:
                print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)

    training = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_add_ops = 0
    total_mul_ops = 0    
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_add_ops += m.total_add_ops
        total_mul_ops += m.total_mul_ops
        total_params += m.total_params

    total_add_ops = total_add_ops.item()
    total_mul_ops = total_mul_ops.item()
    total_params = total_params.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()

    # remove temporal buffers
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if "total_add_ops" in m._buffers:
            m._buffers.pop("total_add_ops")
        if "total_mul_ops" in m._buffers:
            m._buffers.pop("total_mul_ops")
        if "total_params" in m._buffers:
            m._buffers.pop("total_params")

    return total_add_ops, total_mul_ops, total_params