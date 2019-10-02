import os
import time
import argparse
import shutil
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from thop import profile

from Utils import *
from Models import *


def parse_args():
    parser = argparse.ArgumentParser(description='micronet train script')
    parser.add_argument('--model', default='micronet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='name of the dataset to train')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes in dataset')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--which_gpu', default='cuda:1', type=str, help='which GPU to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--mini_batch_size', default=128, type=int, help='mini batch size size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='cos', type=str, help='lr scheduler (cos/step)')
    parser.add_argument('--n_epoch', default=800, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    parser.add_argument('--input_regularize', default='cutmix', type=str, help='input regularization')
    parser.add_argument('--label_regularize', default='crossEntropy',type=str, help='label regularization')
    parser.add_argument('--augmentation', default='FastAuto', type=str, help='data augmentation')
    parser.add_argument('--name', default='micronet_ver10', type=str, help='save file name')
    parser.add_argument('--ortho_lr', default=0.01, type=float, help='orthogonal lr')
    parser.add_argument('--prune_rate', default=30., type=float, help='prune_rate')

    return parser.parse_args()

def get_model():
    print('=> Building model..')
    if args.dataset =='CIFAR100':
        if args.model == 'micronet':
            net = MicroNet(num_classes = args.num_classes, add_se = True, Activation = 'HSwish')
            if args.lr_type == 'cos':
                net.set_config(batch_size = args.batch_size, momentum = args.momentum, lr = args.lr, num_epochs =int(args.n_epoch//4), weight_decay = args.wd, device = args.which_gpu)
            else:
                net.set_config(batch_size = args.batch_size, momentum = args.momentum, lr = args.lr, num_epochs =int(args.n_epoch//4), weight_decay = args.wd, device = args.which_gpu)
        elif args.model == 'micronet_prune':
            net = MicroNet_Prune(num_classes = args.num_classes, add_se = True, Activation = 'HSwish')
            if args.lr_type == 'cos':
                net.set_config(batch_size = args.batch_size, momentum = args.momentum, lr = args.lr, num_epochs =int(args.n_epoch//4), weight_decay = args.wd, device = args.which_gpu)
            else:
                net.set_config(batch_size = args.batch_size, momentum = args.momentum, lr = args.lr, num_epochs =int(args.n_epoch//4), weight_decay = args.wd, device = args.which_gpu)
    elif args.dataset == 'ImageNet':
        if args.model == 'micronet':
            net = MicroNet_imagenet(num_classes = args.num_classes, add_se = True, Activation = 'HSwish')
            if args.lr_type == 'cos':
                net.set_config(batch_size = args.batch_size, momentum = args.momentum, lr = args.lr, num_epochs =int(args.n_epoch), weight_decay = args.wd, device = args.which_gpu)
            else:
                net.set_config(batch_size = args.batch_size, momentum = args.momentum, lr = args.lr, num_epochs =int(args.n_epoch), weight_decay = args.wd, device = args.which_gpu)
    else:
        raise NotImplementedError
    
    
    return net.to(net.device)

def save_checkpoints(state):
    torch.save(state, './Checkpoint/' + args.name + '.t7')
    print('save Checkpoint/' + args.name + '.t7')
    
    
def iterative_train(net, train_loader, test_loader, args, name = 'progres_ver3'):
    #1
    train_losses1, train_accuracy1, test_losses1, test_accuracy1, best_model_wts1 = train_16bit(net, dataloader=train_loader, test_loader=test_loader, lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr)
    torch.save(best_model_wts1, './Checkpoint/' + name + '.t7')
    #2
    train_loader, _, _ = transform_data_set(args.dataset, batch_size = args.batch_size, augmentation = args.augmentation)
    net.load_state_dict(best_model_wts1)
    train_losses2, train_accuracy2, test_losses2, test_accuracy2, best_model_wts2 = train_16bit(net, dataloader=train_loader, test_loader=test_loader, lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr)
    torch.save(best_model_wts2, './Checkpoint/' + name + '.t7')
    #3
    train_loader, _, _ = transform_data_set(args.dataset, batch_size = args.batch_size, augmentation = args.augmentation)
    net.load_state_dict(best_model_wts2)
    train_losses3, train_accuracy3, test_losses3, test_accuracy3, best_model_wts3 = train_16bit(net, dataloader=train_loader, test_loader=test_loader, lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr)
    torch.save(best_model_wts3, './Checkpoint/' + name + '.t7')
    #4
    train_loader, _, _ = transform_data_set(args.dataset, batch_size = args.batch_size, augmentation = args.augmentation)
    net.load_state_dict(best_model_wts3)
    train_losses4, train_accuracy4, test_losses4, test_accuracy4, best_model_wts4 = train_16bit(net, dataloader=train_loader, test_loader=test_loader, lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr)

    train_losses = train_losses1 + train_losses2 + train_losses3 + train_losses4
    train_accuracy = train_accuracy1 + train_accuracy2 + train_accuracy3 + train_accuracy4
    test_losses = test_losses1 + test_losses2 + test_losses3 + test_losses4
    test_accuracy = test_accuracy1 + test_accuracy2 + test_accuracy3 + test_accuracy4
    
    return train_losses, train_accuracy, test_losses, test_accuracy, best_model_wts1, best_model_wts2, best_model_wts3, best_model_wts4

def iterative_prune_train(net, train_loader, test_loader, checkpoint, args, name = 'progres_ver3'):
    #1
    train_losses1, train_accuracy1, test_losses1, test_accuracy1, best_model_wts1 = train_prune_16bit(net, dataloader=train_loader, test_loader=test_loader, best_model_wts_init = checkpoint, lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr, prune_rate = args.prune_rate / 4 )
    torch.save(best_model_wts1, './Checkpoint/' + name + '.t7')
    #2
    train_loader, _, _ = transform_data_set(args.dataset, batch_size = args.batch_size, augmentation = args.augmentation)
    net.load_state_dict(best_model_wts1)
    train_losses2, train_accuracy2, test_losses2, test_accuracy2, best_model_wts2 = train_prune_16bit(net, dataloader=train_loader, test_loader=test_loader, best_model_wts_init = best_model_wts1,lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr, prune_rate = args.prune_rate * 2 / 4)
    torch.save(best_model_wts2, './Checkpoint/' + name + '.t7')
    #3
    train_loader, _, _ = transform_data_set(args.dataset, batch_size = args.batch_size, augmentation = args.augmentation)
    net.load_state_dict(best_model_wts2)
    train_losses3, train_accuracy3, test_losses3, test_accuracy3, best_model_wts3 = train_prune_16bit(net, dataloader=train_loader, test_loader=test_loader, best_model_wts_init = best_model_wts2,lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr, prune_rate = args.prune_rate * 3 / 4)
    torch.save(best_model_wts3, './Checkpoint/' + name + '.t7')
    #4
    train_loader, _, _ = transform_data_set(args.dataset, batch_size = args.batch_size, augmentation = args.augmentation)
    net.load_state_dict(best_model_wts3)
    train_losses4, train_accuracy4, test_losses4, test_accuracy4, best_model_wts4 = train_prune_16bit(net, dataloader=train_loader, test_loader=test_loader, best_model_wts_init = best_model_wts3,lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr, prune_rate = args.prune_rate)

    train_losses = train_losses1 + train_losses2 + train_losses3 + train_losses4
    train_accuracy = train_accuracy1 + train_accuracy2 + train_accuracy3 + train_accuracy4
    test_losses = test_losses1 + test_losses2 + test_losses3 + test_losses4
    test_accuracy = test_accuracy1 + test_accuracy2 + test_accuracy3 + test_accuracy4
    
    return train_losses, train_accuracy, test_losses, test_accuracy, best_model_wts1, best_model_wts2, best_model_wts3, best_model_wts4
 
        
if __name__ == '__main__':
    args = parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    print('=> Preparing data..')    
    train_loader, test_loader, num_classes = transform_data_set(args.dataset, batch_size = args.batch_size, augmentation = args.augmentation)
    
    net = get_model()
    
    #convert to half precision
    net.half()  
    for layer in net.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
    
    best_model_wts_init = copy.deepcopy(net.state_dict())
    
    if args.dataset =='CIFAR100':
        input = torch.randn(1, 3, 32, 32).type(torch.HalfTensor).to(net.device)
        flops, params = profile(net, inputs=(input, ))
    if args.dataset =='ImageNet':
        input = torch.randn(1, 3, 224, 224).type(torch.HalfTensor).to(net.device)
        flops, params = profile(net, inputs=(input, ))
                
    print('flops: {}, params: {}'.format(flops, params))
    print('start ' + args.name + '.t7')
    if args.dataset == 'CIFAR100':
        if args.model == 'micronet':
            train_losses, train_accuracy, test_losses, test_accuracy, best_model_wts1, best_model_wts2, best_model_wts3, best_model_wts4 = iterative_train(net, train_loader, test_loader, args, name = 'progres_ver10')
        elif args.model == 'micronet_prune':
            checkpoint = torch.load('./Checkpoint/micronet_ver2.t7')
            net.load_state_dict(checkpoint['net4'])
            train_losses, train_accuracy, test_losses, test_accuracy, best_model_wts1, best_model_wts2, best_model_wts3, best_model_wts4 = iterative_prune_train(net, train_loader, test_loader, checkpoint['net_init'], args, name = 'progres_ver5')
    elif args.dataset == 'ImageNet':
        train_losses, train_accuracy, test_losses, test_accuracy, best_model_wts = train_image_16bit(net, train_loader, test_loader, args, lr_type = args.lr_type, input_regularize = args.input_regularize, label_regularize = args.label_regularize, ortho = True, ortho_lr = args.ortho_lr)
        
    
    state = {}
    state['net_init'] = best_model_wts_init
    if args.dataset == 'CIFAR100':
        state['net1'] = best_model_wts1
        state['net2'] = best_model_wts2
        state['net3'] = best_model_wts3
        state['net4'] = best_model_wts4
    elif args.dataset == 'ImageNet':
        state['net'] = best_model_wts
    
    state['train_losses'] = train_losses
    state['train_accuracy'] = train_accuracy
    state['test_losses'] = test_losses
    state['test_accuracy'] = test_accuracy
    state['flops_params'] = (flops, params)
    state['Score'] = flops / 1170000000 + params/6900000
    
    save_checkpoints(state)