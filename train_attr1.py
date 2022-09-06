# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 20:18:31 2022
@author: linxi
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data_attr import MotorAttribute
from models import DGCNN_net
from utils import mean_loss, IOStream, normalize_data
from torch.utils.tensorboard import SummaryWriter




def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name)
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/model'):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/model')


def train(args, io):
    train_data = MotorAttribute(root_dir=args.root, csv_file='motor_attr.csv', mask_file='attr_mask.csv',
                                split='train', test_area=args.validation_symbol)
    train_dataloader = DataLoader(train_data, num_workers=8, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
    test_data = MotorAttribute(root_dir=args.root, csv_file='motor_attr.csv', mask_file='attr_mask.csv',
                               split='test', test_area=args.validation_symbol)
    test_dataloader = DataLoader(test_data, num_workers=8, batch_size=args.test_batch_size,
                                 shuffle=True, drop_last=False)
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    model = DGCNN_net().to(device)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPU!")
    
    if args.opt == 'sgd':
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, 
                        weight_decay=1e-4)
    elif args.opt == 'adam':
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'adamw':
        print("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=60, gamma=0.2)
    
    print("Starting from scratch!")
    
    criterion = mean_loss
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        for pc, ty, attr, num, mask in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9):
            pc, ty, attr, num, mask = pc.to(device), ty.to(device), attr.to(device), num.to(device), mask.to(device)
            pc = normalize_data(pc)
            data = pc.permute(0, 2, 1)
            batch_size = data.size()[0]
            t = ty.reshape(-1)
            type_one_hot = F.one_hot(t.long(), num_classes=5)
            num_one_hot = F.one_hot(num.reshape(-1).long(), num_classes=7)
            opt.zero_grad()
            pred_attr = model(data.float(), type_one_hot.float(), num_one_hot.float())
            loss = criterion(pred_attr.view(-1, 28), attr.view(-1, 28), mask=mask)
            loss.backward()
            opt.step()
            count += batch_size
            train_loss += loss.item() * batch_size
        
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
                    
        outstr = 'Train %d, Loss: %.6f' % (epoch, train_loss*1.0/count)
        io.cprint(outstr)
        writer.add_scalar('learning rate/lr', opt.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train loss', train_loss*1.0/count, epoch)
        
        ####################
        # Validation
        ####################
        val_loss = 0.0
        count = 0.0
        model.eval()
        for pc, ty, attr, num, m in tqdm(test_dataloader, total=len(test_dataloader), smoothing=0.9):
            pc, ty, attr, num, m = pc.to(device), ty.to(device), attr.to(device), num.to(device), m.to(device)
            pc = normalize_data(pc)
            data = pc.permute(0, 2, 1)
            batch_size = data.size()[0]
            t = ty.reshape(-1)
            type_one_hot = F.one_hot(t.long(), num_classes=5)
            num_one_hot = F.one_hot(num.reshape(-1).long(), num_classes=7)
            pred_attr = model(data.float(), type_one_hot.float(), num_one_hot.float())
            loss = criterion(pred_attr.view(-1, 28), attr.view(-1, 28), mask=m)
            count += batch_size
            val_loss += loss.item() * batch_size
        
        outstr = 'Test %d, Loss: %.6f' % (epoch, val_loss*1.0/count)
        io.cprint(outstr)

        writer.add_scalar('Loss/test loss', val_loss*1.0/count, epoch)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi Attributes Regression')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--change', type=str, default='hh', metavar='N',
                        help='explict parameters in experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--root', type=str, metavar='N',default='E:\\dataset1000',
                        help='folder of dataset')
    parser.add_argument('--csv', type=str, default='E:\\data\\motor_attr.csv',
                        help='moter attributes')
    parser.add_argument('--mask', type=str, default='E:\\data\\attr_mask',
                        help='attributes mask')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--opt', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'],
                        help='optimizer to use, [SGD, Adam, AdamW]')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=32, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--validation_symbol', type=str, default='Validation', 
                        help='Which datablocks to use for validation')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    
    _init_()

    writer = SummaryWriter('outputs/' + args.model + '/' + args.exp_name + '/' + args.change)
    
    io = IOStream('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/result.log')
    io.cprint(str(args))
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')
    
    if not args.eval:
        train(args, io)