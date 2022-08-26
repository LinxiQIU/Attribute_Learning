# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:01:12 2022

@author: linxi Qiu
"""


import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data_attr import MotorAttribute
from models import DGCNN_cls
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name)
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/model'):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/model')



def train(args, io):
    train_data = MotorAttribute(root_dir=args.root, csv_file='motor_attr.csv', 
                                split='train', test_area=args.validation_symbol)
    train_dataloader = DataLoader(train_data, num_workers=8, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True)
    test_data = MotorAttribute(root_dir=args.root, csv_file='motor_attr.csv',
                               split='test', test_area=args.validation_symbol)
    test_dataloader = DataLoader(test_data, num_workers=8, batch_size=args.test_batch_size,
                                 shuffle=True, drop_last=False)
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2_cls(args).to(device)
    else:
        raise Exception('Not implemented')
    print(str(model))
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
        
    criterion = cal_loss  
    best_val_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0
        model.train()
        train_pred = []
        train_true = []
        for sample in train_dataloader:
            data, label = sample['point'], sample['type']
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data.float())
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
                    
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))    
            
        io.cprint(outstr)
        writer.add_scalar('learning rate/lr', opt.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train loss', train_loss*1.0/count, epoch)
        writer.add_scalar('Accuracy/train acc', metrics.accuracy_score(train_true, train_pred), epoch)
        writer.add_scalar('Average Accuracy/train avg acc', metrics.balanced_accuracy_score(train_true, train_pred), epoch)
        ####################
        # Validation
        ####################
        val_loss = 0.0
        count = 0.0
        model.eval()
        val_pred = []
        val_true = []
        for sample in test_dataloader:
            data, label = sample['point'], sample['type']
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data.float())
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            val_loss += loss.item() * batch_size
            val_true.append(label.cpu().numpy())
            val_pred.append(preds.detach().cpu().numpy())
        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        val_acc = metrics.accuracy_score(val_true, val_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(val_true, val_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              val_loss*1.0/count,
                                                                              val_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        writer.add_scalar('Loss/val loss', val_loss*1.0/count, epoch)
        writer.add_scalar('Accuracy/val acc', val_acc, epoch)
        writer.add_scalar('Average Accuracy/val avg acc', avg_per_class_acc, epoch)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'outputs/%s/%s/%s/model.t7' % (args.model, args.exp_name, args.change))


# def test(args, io):
#     test_dataloader = DataLoader(MotorDataset(root=args.root, split='train', 
#                                            num_points=args.num_points, 
#                                            test_area=args.validation_symbol),
#                               num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)

#     device = torch.device("cuda" if args.cuda else "cpu")

#     if args.model == 'dgcnn':
#         model = DGCNN_cls(args).to(device)

#     model = nn.DataParallel(model)
#     model.load_state_dict(torch.load(args.model_path))
#     model = model.eval()
#     test_acc = 0.0
#     count = 0.0
#     test_true = []
#     test_pred = []

#     for data, label in test_loader:
#         data, label = data.to(device), label.to(device).squeeze()
#         data = data.permute(0, 2, 1)
#         batch_size = data.size()[0]
#         logits = model(data.float())
#         preds = logits.max(dim=1)[1]
#         test_true.append(label.cpu().numpy())
#         test_pred.append(preds.detach().cpu().numpy())
#     test_true = np.concatenate(test_true)
#     test_pred = np.concatenate(test_pred)
#     test_acc = metrics.accuracy_score(test_true, test_pred)
#     avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

#     outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
#     io.cprint(outstr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--change', type=str, default='hh', metavar='N',
                        help='explict parameters in experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'pointnet2'],
                        help='Model to use, [dgcnn, pct]')
    parser.add_argument('--root', type=str, metavar='N',default='E:\\dataset',
                        help='folder of dataset')
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
    # else:
    #     test(args, io)
        
        
        
        
        
        