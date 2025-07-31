
import os
import datetime

import numpy

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .data_transform import GaussianBlur, get_moco_base_augmentation

data_folder = '../dataset' # for greene,  '../dataset' for local


def get_dataloader(args):

    if args.dset in ["cifar10", "cifar100"]:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        if (args.aug is None) or (args.aug == 'null'):
            transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        elif args.aug == 'pc':  # padded crop
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize])
        elif args.aug == 'rs':  # resized crop
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
                transforms.ToTensor(),
                normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if args.dset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../dataset', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
            )
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../dataset', train=False, download=True, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
            )
        elif args.dset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../dataset', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
            )
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../dataset', train=False, download=True, transform=transform_test),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
            )

    if args.dset == 'stl10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2434, 0.2615])
        transform = transforms.Compose([
            # transforms.RandomCrop(96, padding=4), # for stl10
            transforms.ToTensor(),
            normalize
        ])
        test_tranform = transform
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10('data', split='train', download=True, transform=transform),
            batch_size=args.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, persistent_workers=True
            )
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10('data', split='test', download=True, transform=test_tranform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
            )
    
    elif args.dset == 'mnist':
        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
        trainset = datasets.MNIST(root='../dataset', train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        valset = datasets.MNIST(root='../dataset', train=False, download=True, transform=transform)
        test_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    elif args.dset == 'tinyi': # image_size:64 x 64
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.ToTensor(),
                                        normalize,
                                        ])
        test_transform = transform
        
        train_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'train'), transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_folder, 'tiny-imagenet-200', 'val/organized_val'), test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader