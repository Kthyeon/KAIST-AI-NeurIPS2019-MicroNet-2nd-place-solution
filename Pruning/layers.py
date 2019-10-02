import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Pruning.utils import to_var
from torch.nn.parameter import Parameter

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.device = device
    
    def set_mask(self, mask):
        tmp = copy.deepcopy(mask)
        tmp = torch.tensor(tmp).to(self.device)
        self.register_buffer('mask', tmp)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        return to_var(self.mask, self.device, requires_grad=False)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            weight = self.weight * mask_var
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
        
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        self.device = device
    def set_mask(self, mask):
        tmp = copy.deepcopy(mask)
        tmp = torch.tensor(tmp).to(self.device)
        self.register_buffer('mask', tmp)
        mask_var = self.get_mask()
        # print('mask shape: {}'.format(self.mask.data.size()))
        # print('weight shape {}'.format(self.weight.data.size()))
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, self.device, requires_grad=False)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            # print(self.weight)
            # print(self.mask)
            # print('weight/mask id: {} {}'.format(self.weight.get_device(), mask_var.get_device()))
            self.weight.data = self.weight.data * mask_var.data
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.device = device
    
    def set_mask(self, mask):
        tmp = copy.deepcopy(mask)
        tmp = torch.tensor(tmp).to(self.device)
        self.register_buffer('mask', tmp)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        return to_var(self.mask, self.device, requires_grad=False)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            self.weight.data = self.weight.data * mask_var.data
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
class ClipBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, device, eps = 1e-5, momentum = 0.01, affine=True, track_running_stats = True):
        super(ClipBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.device = device
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight.data = Parameter(torch.Tensor(num_features)).to(self.device)
            self.bias.data = Parameter(torch.Tensor(num_features)).to(self.device)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features).to(self.device))
            self.register_buffer('running_var', torch.ones(num_features).to(self.device))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long).to(self.device))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        self.weight.data.clamp_(min=0.8, max=1.2)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
