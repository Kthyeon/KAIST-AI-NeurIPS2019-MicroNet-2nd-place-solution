import torch
import torch.nn
from torch import cuda
from torchvision import datasets, transforms
import os
import ast
from thop import profile
from torch.utils.data import DataLoader, sampler

from Utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from Utils.cifar100_fast_aug import fa_shake26_2x96d_cifar100, fa_wresnet40x2_cifar100_r5, fa_wresnet40x2_cifar100, fa_wresnet28x10_cifar100
from Utils.fast_augmentations import *


def transform_data_set(datatype, batch_size = 96, augmentation = 'FastAuto'):
    """
    ---Preprocessing
    RandomCrop => RandomHorizontalFlip => ToTensor => Normalize

    --datatype
    CIFAR10, CIFAR100, svhn, imagenet

    """
    if datatype == 'CIFAR10':
        transform_train = transforms.Compose([
                                                        transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                    ])
        transform_test = transforms.Compose(
                                                        [transforms.ToTensor(),
                                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
        trainset = datasets.CIFAR10(root='./Data/CIFAR10', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, shuffle=True, num_workers=16)
        
        testset = datasets.CIFAR10(root='./Data/CIFAR10', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=16)
        num_classes = 10
        
        return train_loader, test_loader, num_classes

    elif datatype == 'CIFAR100':
        if augmentation == 'cutout':
            print('|| Prepare dataset with cutout ||')
            transform_train = transforms.Compose([
                                                        transforms.RandomCrop(32, padding=4, fill = 128),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        Cutout(n_holes=1, length=16),
                                                        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762)),
                                                    ])
            trainset = datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=True, transform=transform_train)
        elif augmentation == 'FastAuto':
            print('|| Prepare dataset with FastAutoAugmentation ||')
            transform_train1 = transforms.Compose([
                                                        transforms.RandomCrop(32, padding=4, fill = 128),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762)),
                                                    ])
            transform_train2 = transforms.Compose([
                                                        transforms.RandomCrop(32, padding=4, fill = 128),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762)),
                                                    ])
            transform_train3 = transforms.Compose([
                                                        transforms.RandomCrop(32, padding=4, fill = 128),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762)),
                                                    ])
            transform_train4 = transforms.Compose([
                                                        transforms.RandomCrop(32, padding=4, fill = 128),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762)),
                                                    ])

            transform_train1.transforms.insert(0, Augmentation(fa_wresnet28x10_cifar100()))
            transform_train2.transforms.insert(0, Augmentation(fa_shake26_2x96d_cifar100()))
            transform_train3.transforms.insert(0, Augmentation(fa_wresnet40x2_cifar100()))
            transform_train4.transforms.insert(0, Augmentation(fa_wresnet40x2_cifar100_r5()))

            trainset = datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=True, transform=transform_train1)
            trainset += datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=False, transform=transform_train2)
            trainset += datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=False, transform=transform_train3)
            trainset += datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=False, transform=transform_train4)
        elif augmentation == 'Auto':
            print('|| Prepare dataset with AutoAugmentation ||')
            transform_train = transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4, fill = 128),
                                                    transforms.RandomHorizontalFlip(),CIFAR10Policy(),
                                                    transforms.ToTensor(),
                                                    Cutout(n_holes=1, length=16),
                                                    transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762)),
                                                ])
            trainset = datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=False, transform=transform_train)
        else:
            print('|| Prepare dataset with standard Augmentation ||')
            transform_train = transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4, fill = 128),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762)),
                                                ])
            trainset = datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=True, transform=transform_train)
            trainset += datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=False, transform=transform_train)
            trainset += datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=False, transform=transform_train)
            trainset += datasets.CIFAR100(root='./Data/CIFAR100', train=True, download=False, transform=transform_train)

            
        transform_test = transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5071, 0.4865, 0.4409) , (0.2673, 0.2564, 0.2762))])
        
        
        testset = datasets.CIFAR100(root='./Data/CIFAR100', train=False, download=False, transform=transform_test)
            
        train_loader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        num_classes = 100
        
        return train_loader, test_loader, num_classes
    elif datatype == 'ImageNet':
        traindir = '/home/taehyeon/ImageNet/Data/train/'
        validdir = '/home/taehyeon/ImageNet/Data/val/'
        if augmentation == 'Auto':
            print('|| Prepare dataset with AutoAugmentation ||')
            transform_train =transforms.Compose(
                        [transforms.RandomResizedCrop(224), 
                         transforms.RandomHorizontalFlip(), ImageNetPolicy(), 
                         transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
            
            transform_test = transforms.Compose([
                                transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
            
            trainset = datasets.ImageFolder(root=traindir, transform=transform_train)
            testset = datasets.ImageFolder(root=validdir, transform=transform_test)
            
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4,
           pin_memory=True)
            test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4,
           pin_memory=True)
            
            num_classes = 1000
            
            return train_loader, test_loader, num_classes