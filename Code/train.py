import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from dataset import get_dataloaders
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from nni.compression.torch import AGP_Pruner, Pruner
from nni.compression.torch.pruning.weight_masker import WeightMasker
from nni.compression.torch.pruning.structured_pruning import ActivationFilterPrunerMasker, StructuredWeightMasker

import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm


def draw_weights(weights, index):
    print(weights.shape)
    weights = weights.cpu().numpy()
    for i in range(len(weights)):
        fig = plt.figure()
        plt.bar(np.arange(len(weights[i])), weights[i])
        #plt.show()
        fig.savefig("scaling/scaling" + str(index))
        print("min = ", min(weights[i]))
        print("# of zeros = ", np.count_nonzero(weights[i] == 0))

class MyMasker(StructuredWeightMasker):
    def calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        weight = wrapper.module.weight.data
        bias = None
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            bias = wrapper.module.bias.data

        if wrapper.weight_mask is None:
            mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        else:
            mask_weight = wrapper.weight_mask.clone()
        if bias is not None:
            if wrapper.bias_mask is None:
                mask_bias = torch.ones(bias.size()).type_as(bias).detach()
            else:
                mask_bias = wrapper.bias_mask.clone()
        else:
            mask_bias = None
        mask = {'weight_mask': mask_weight, 'bias_mask': mask_bias}

        filters = weight.size(0)
        num_prune = int(filters * sparsity)
        if filters < 2 or num_prune < 1:
            return mask
        # weight*mask_weight: apply base mask for iterative pruning
        return self.get_mask(mask, weight*mask_weight, num_prune, wrapper, wrapper_idx)

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx):
        print(activation[list(activation.keys())[wrapper_idx]])
        print(activation[list(activation.keys())[wrapper_idx]].shape)
        mask = torch.mean(activation[list(activation.keys())[wrapper_idx]], dim  = 0, keepdims = True)
        draw_weights(mask, wrapper_idx)
        threshold =  torch.topk(mask[0], k = num_prune, dim = 0, largest=False)[0].max()
        mask_weight = torch.gt(mask[0], threshold)[:, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(mask[0], threshold).type_as(weight).detach() if base_mask['bias_mask'] is not None else None
        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}



class MyPruner(Pruner):
    def __init__(self, model, config_list, optimizer):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        # construct a weight masker instance
        self.masker = MyMasker(model, self)

    def calc_mask(self, wrapper, wrapper_idx=None):
        sparsity = wrapper.config['sparsity']
        if wrapper.if_calculated:
            # Already pruned, do not prune again as a one-shot pruner
            return None
        else:
            # call your masker to actually calcuate the mask for this layer
            masks = self.masker.calc_mask(sparsity=sparsity, wrapper=wrapper, wrapper_idx=wrapper_idx)
            wrapper.if_calculated = True
            return masks

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def Average(lst): 
    return sum(lst) / len(lst)


def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    inference_time=[]

    # calculate FLOPS:
    from thop import profile

    macs, params = profile(net, inputs=(torch.randn(1, 3, settings.IMG_SIZE, settings.IMG_SIZE).cuda(), ))
    print("macs = ", macs)
    print("params = ", params)

    for (images, labels) in tqdm(test_loader):
        with torch.no_grad():
            start = timeit.default_timer()
            images = Variable(images)
            labels = Variable(labels)
            
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            stop = timeit.default_timer()
            inference_time.append(stop-start)
            loss = loss_function(outputs, labels)
            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    print("FPS = ", 1/Average(inference_time))
        

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

num_classes = {'dogs': 120, 'tiny-imagenet': 200, 'cifar100': 100, 'cifar10': 10, 'caltech': 257, 'imagenet': 1000}
if __name__ == '__main__':
    # #config for pruner
    # config_list = [{
    # 'initial_sparsity': 0.0,
    # 'final_sparsity': 0.8,
    # 'start_epoch': 0,
    # 'end_epoch': 200,
    # 'frequency': 1,
    # 'op_types': ['Conv2d']
    # }]
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-weights', type=str, default='', help='the weights file you want to load')
    parser.add_argument('-data', type=str, default='dogs', help='the weights file you want to load')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu, num_classes = num_classes[args.data])
    # print number of paramters
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("number of network paramters are ", pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("number of network Trainable paramters are ", pytorch_total_params)
       

    if args.weights != '':
        net.load_state_dict(torch.load(args.weights), args.gpu)
        print('loaded checkpoint')

    dataloaders = get_dataloaders(args.b, args.data)
    #data preprocessing:aset)
    #data preprocessing:
    training_loader = dataloaders['train']
    
    test_loader = dataloaders['val']
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #"""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=9,
                                                           verbose=True, threshold=0.001, threshold_mode='rel',
                                                           cooldown=0, min_lr=1e-6, eps=1e-08)
    #"""
    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    # #pruner
    # pruner = AGP_Pruner(net, config_list, optimizer, pruning_algorithm='l1')
    # pruner.compress()
    # activation = {}
    # conv_layers = []
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook
    # for name, module in net.named_modules():
    #     # print(name)
    #     if len(name) > 1 and name[-1] == '3' and 'excitation' in name:
    #         module.register_forward_hook(get_activation(name))
    #     elif len(name) > 1 and (name[-13 : ] == "bottle_neck.5" or name[-10:] == 'residual.6'):
    #         conv_layers.append(name)
    
    # print(conv_layers)
    # # print(net)
    # config_list = [{
    # 'sparsity': 0.5,
    # 'op_types': ['Conv2d'],
    # 'op_names': conv_layers}]

    # print(net)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, settings.IMG_SIZE, settings.IMG_SIZE).cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))
    # pruner = MyPruner(net, config_list, optimizer)

    #acc = eval_training(0)
    # pruner.compress()
    # pruner.export_model(model_path='test.pth', mask_path='test.pth')
    # exit()
    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        #update pruner
        # pruner.update_epoch(epoch)
        #if epoch > args.warm:
        #    train_scheduler.step(epoch)


        train(epoch)
        acc = eval_training(epoch)

        if epoch > args.warm:
            scheduler.step(acc)

        #start to save best performance model after learning rate decay to 0.01 
        if best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    # pruner.export_model(model_path='model_l1_freq1.pth', mask_path='mask_l1_freq1.pth')
    writer.close()
