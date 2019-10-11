'''MicroNet

Option
1. Activation : ReLU, HSwish
2. Squeeze - and - Excitation ratio

    No pruning version
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
import math
import numpy as np


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    Parameters:
    ----------
    inplace : bool
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace
        self.relu = nn.ReLU6(inplace = self.inplace)

    def forward(self, x):
        return x * self.relu(x + 3.0) / 6.0

class MicroBlock(nn.Module):
    '''expand + depthwise + pointwise
    Activation : ReLU or HSwish
    
    '''
    def __init__(self, in_planes, out_planes, expansion, stride, device, add_se = False, Activation = 'ReLU'):
        super(MicroBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        planes = int(expansion * in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.add_se = add_se
        
        if Activation == 'HSwish':
            self.act1 = HSwish()
            self.act2 = HSwish()
            if self.add_se:
                self.act_se = HSwish()
        else:
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            if self.add_se:
                self.act_se = nn.ReLU()
            
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.01)
            )
        
        # SE layers
        
        if self.add_se:
            self.avg_se = nn.AdaptiveAvgPool2d(1)
            number = int(out_planes*0.25)
            self.fc1 = nn.Conv2d(out_planes, number, kernel_size=1)
            self.fc2 = nn.Conv2d(number, out_planes, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
            
        

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        #out = (out - 1/6) / np.sqrt(11/36)
        out = self.act2(self.bn2(self.conv2(out)))
        #out = (out - 1/6) / np.sqrt(11/36)
        out = self.bn3(self.conv3(out))
        
        # Squeeze-Excitation
        if self.add_se:
            w = self.avg_se(out)
            w = self.act_se(self.fc1(w))
            w = self.sigmoid(self.fc2(w))

            out = out * w + self.shortcut(x) if self.stride==1 else out
            return out
        
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MicroNet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    def __init__(self, ver = 'ver2', num_classes=100, wide_factor = 1, depth_factor =1, add_se = True, Activation = 'HSwish'):
        super(MicroNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        '''
        wide_factor: channel을 늘리는 비율
        depth_factor: depth를 늘리는 비율
        '''
        if ver == 'ver2':
            self.cfg = [[2.5, 20, 2, 1],
                        [2.5, 36, 1, 2],
                        [2.5, 36, 1, 1],
                        [2.5, 56, 3, 1],
                        [2.5, 80, 1, 2],
                        [2.5, 80, 4, 1],
                        [2.5, 88, 1, 2],
                        [2.5, 96, 2, 1],
                        [2.5, 114, 1, 1]]
        else:
            self.cfg = [[3, 16, 2, 1],
                        [3, 32, 1, 2],
                        [3, 32, 1, 1],
                        [3, 48, 3, 1],
                        [3, 72, 1, 2],
                        [3, 72, 4, 1],
                        [3, 80, 1, 2],
                        [3, 88, 2, 1],
                        [3, 106, 1, 1]]

        #reconstruct structure config
        self.change_cfg(wide_factor, depth_factor)
        #make train recipe
        self.set_config(batch_size = 128, momentum = 0.9, lr = 0.1, num_epochs =200, criterion = nn.CrossEntropyLoss(), weight_decay = 1e-5, gamma = 0.1, milestones = [100, 150], device = 'cuda:0' if cuda.is_available() else 'cpu', nesterov = True)
        
        #construct network
        self.add_se = add_se
        self.Activation = Activation
        self.input_channel = 32
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, self.input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.input_channel, momentum=0.01)
        self.bn1 = nn.BatchNorm2d(self.input_channel, momentum=0.01)
        self.layers = self._make_layers(in_planes=self.input_channel)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.cfg[-1][1], self.num_classes)
        
        if self.Activation == 'HSwish':
            self.stem_act = HSwish()
        else:
            self.stem_act = nn.ReLU()
        
        #initialize the parameters
        self.reset_parameters()
        
        #initialize the parameters
        self.reset_custom_parameters()
        
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    #m.bias = nn.Parameter(torch.zeros(m.running_var.shape, dtype=m.running_var.dtype, device=m.running_var.device), requires_grad=False)
                else:
                    if m.weight is None:
                        m.weight = nn.Parameter(torch.ones(m.running_var.shape, dtype=m.running_var.dtype, device=m.running_var.device), requires_grad=False)
                    if m.bias is None:
                        m.bias = nn.Parameter(torch.zeros(m.running_var.shape, dtype=m.running_var.dtype, device=m.running_var.device), requires_grad=False)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def reset_custom_parameters(self):
        for name, module in self.named_children():
            if 'layer' in name:
                if 'conv1' in name:
                    n = module.shape[1]
                    module.data.normal_(0, 0.65 * math.sqrt(1. / n))
                elif 'conv2' in name:
                    n = module.shape[0]
                    module.data.normal_(0, 9.37 * math.sqrt(2. / n))
                elif 'conv3' in name:
                    n = module.shape[1]
                    module.data.normal_(0, 1.18 * math.sqrt(1. / n))
                    
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(MicroBlock(in_planes, out_planes, expansion, stride, self.device, self.add_se, self.Activation))
                in_planes = out_planes
        return nn.Sequential(*layers)
    
    def change_cfg(self, wide_factor, depth_factor):
        for i in range(len(self.cfg)):
            self.cfg[i][1] = int(self.cfg[i][1] * wide_factor)
            if self.cfg[i][3] ==1:
                self.cfg[i][2] = int(self.cfg[i][2] * depth_factor)
    
    
    def set_config(self, batch_size, momentum, lr, num_epochs, device, weight_decay, gamma = 0.1, milestones = [100,150], nesterov = True, criterion = nn.CrossEntropyLoss()):
        self.batch_size = batch_size
        self.momentum = momentum
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.milestones = milestones
        self.device = device
        self.nesterov = nesterov
        
    def forward(self, x):
        #stem
        out = self.stem_act(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.linear(self.dropout(out))
            
        return out

    

