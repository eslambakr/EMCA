""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy 
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision

def get_dataloaders(batch_size, dataset):
    print(dataset)
    if dataset == 'dogs':
        image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=135, scale=(0.95, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=128),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(size=128),
            transforms.CenterCrop(size=128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        }   
        all_data = datasets.ImageFolder(root='data/archive/images/Images/')
        torch.manual_seed(42)
        train_data_len = int(len(all_data)*0.8)
        valid_data_len = int((len(all_data) - train_data_len))
        train_data, val_data = torch.utils.data.random_split(all_data, [train_data_len, valid_data_len])
        train_data.dataset.transform = image_transforms['train']
        val_data.dataset.transform = image_transforms['test']
        print(len(train_data), len(val_data))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

        return {'train': train_loader, 'val' :val_loader}

    elif dataset == 'imagenet':
        data_path = "/mnt/4T_2/imagenet_dataset"
        traindir = os.path.join(data_path, 'train')
        valdir = os.path.join(data_path, 'val')
        image_transforms = {
            'train':
                transforms.Compose([transforms.RandomRotation(degrees=15), transforms.ColorJitter(),
                                    transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            'test':
                transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        }
        train_dataset = datasets.ImageFolder(traindir, image_transforms["train"])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=30, pin_memory=True)
        val_dataset = datasets.ImageFolder(valdir, image_transforms["test"])
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=30, pin_memory=True)
        print(len(train_dataset), len(val_dataset))
        return {'train': train_loader, 'val': val_loader}

    elif dataset == 'tiny-imagenet':
        image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=64),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        'test':
        transforms.Compose([
            transforms.Resize(size=64),
            transforms.CenterCrop(size=64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        }
        all_data = datasets.ImageFolder(root='data/tiny-imagenet/tiny-imagenet-200/train')
        torch.manual_seed(42)
        train_data_len = int(len(all_data)*0.8)
        valid_data_len = int((len(all_data) - train_data_len))
        train_data, val_data = torch.utils.data.random_split(all_data, [train_data_len, valid_data_len])
        train_data.dataset.transform = image_transforms['train']
        val_data.dataset.transform = image_transforms['test']
        print(len(train_data), len(val_data))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

        return {'train': train_loader, 'val' :val_loader}

    elif dataset == 'cifar100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        image_transforms = {
        'train':
        transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        }   
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=image_transforms['train'])
        train_loader = torch.utils.data.DataLoader(
            cifar100_training, shuffle=True, batch_size=batch_size)

        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=image_transforms['test'])
        val_loader = torch.utils.data.DataLoader(
            cifar100_test, shuffle=True, batch_size=batch_size)

        return {'train': train_loader, 'val' :val_loader} 

    elif dataset == 'cifar10':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        image_transforms = {
        'train':
        transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        }   
        cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=image_transforms['train'])
        train_loader = torch.utils.data.DataLoader(
            cifar10_training, shuffle=True, batch_size=batch_size)

        cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=image_transforms['test'])
        val_loader = torch.utils.data.DataLoader(
            cifar10_test, shuffle=True, batch_size=batch_size)

        return {'train': train_loader, 'val' :val_loader} 


    elif dataset == 'caltech':
        image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        'test':
        transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        }   
        all_data = datasets.ImageFolder(root='data/caltech/256_ObjectCategories')
        torch.manual_seed(42)
        train_data_len = int(len(all_data)*0.8)
        valid_data_len = int((len(all_data) - train_data_len))
        train_data, val_data = torch.utils.data.random_split(all_data, [train_data_len, valid_data_len])
        train_data.dataset.transform = image_transforms['train']
        val_data.dataset.transform = image_transforms['test']
        print(len(train_data), len(val_data))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

        return {'train': train_loader, 'val' :val_loader}



    else:
        print('This dataset isn\'t supported yet') 
