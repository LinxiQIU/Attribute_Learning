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
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data_attr import MotorAttribute
from models import DGCNN_Core, Attribute, CLS_Semseg, TWO_CLS
from utils import cal_loss, IOStream, normalize_data
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter




def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name)
    if not os.path.exists('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/models'):
        os.makedirs('outputs/' + args.model + '/' + args.exp_name + '/' + args.change + '/models')


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
    
    Head = nn.DataParallel(DGCNN_Core().to(device))
    if args.with_seg is True:
        print('With semseg!')
        Tail1 = nn.DataParallel(CLS_Semseg().to(device))
    else:
        print('No semseg!')
        Tail1 = nn.DataParallel(TWO_CLS().to(device))
    print("Let's use", torch.cuda.device_count(), "GPU!")
    params = list(Head.parameters()) + list(Tail1.parameters()) 
    if args.opt == 'sgd':
        print("Use SGD")        
        opt = optim.SGD(params, lr=args.lr*100, momentum=args.momentum, 
                        weight_decay=1e-4)
    elif args.opt == 'adam':
        print("Use Adam")
        opt = optim.Adam(params, lr=args.lr, weight_decay=1e-2)

    elif args.opt == 'adamw':
        print("Use AdamW")
        opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=60, gamma=0.2)  
    print("Starting from scratch!")    

    criterion1 = cal_loss
    num_class = 7
    best_mse = 100
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0
        total_seen = 0
        total_correct = 0
        Head.train()
        Tail1.train()
        train_pred_cls = []
        train_true_cls = []
        train_pred_num = []
        train_true_num = []
        total_correct_class = [0 for _ in range(num_class)]
        total_iou_deno_class = [0 for _ in range(num_class)]
        for pc, seg, ty, attr, num, mask in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9):
            pc, seg, ty, attr, num, mask = pc.to(device), seg.to(device), ty.to(device), attr.to(device), num.to(device), mask.to(device)
            pc = normalize_data(pc)
            data = pc.permute(0, 2, 1)
            batch_size = data.size()[0]
            num = torch.sub(num, 3)
            opt.zero_grad()
            pointweise, global_feature = Head(data.float())
            if args.with_seg is True:
                pred_seg, pred_ty, pred_num = Tail1(pointweise, global_feature)
                pred_seg = pred_seg.permute(0, 2, 1).contiguous().view(-1, num_class)
                loss_seg = criterion1(pred_seg, seg.view(-1, 1).squeeze())
                loss_cls = criterion1(pred_ty, ty.squeeze())            
                loss_num = criterion1(pred_num, num.squeeze())
                loss = loss_seg + loss_cls + loss_num
                pred_choice = pred_seg.cpu().data.max(1)[1].numpy()
                batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (batch_size * args.num_points)
                for l in range(num_class):
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))    # Intersection
                    total_iou_deno_class[l] += np.sum((pred_choice == l) | (batch_label == l))   # Union
            else:
                pred_ty, pred_num = Tail1(global_feature)
                loss_cls = criterion1(pred_ty, ty.squeeze())            
                loss_num = criterion1(pred_num, num.squeeze())
                loss = loss_cls + loss_num
            loss.backward()
            opt.step()
            logits = pred_ty.max(dim=1)[1]
            train_pred_cls.append(logits.detach().cpu().numpy())
            train_true_cls.append(ty.cpu().numpy())
            cb_num = pred_num.max(dim=1)[1]
            train_pred_num.append(cb_num.detach().cpu().numpy())
            train_true_num.append(num.cpu().numpy())
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
        train_pred_cls = np.concatenate(train_pred_cls)
        train_true_cls = np.concatenate(train_true_cls)
        train_type_cls = accuracy_score(train_true_cls, train_pred_cls)
        train_pred_num = np.concatenate(train_pred_num)
        train_true_num = np.concatenate(train_true_num)
        train_num_acc = accuracy_score(train_true_num, train_pred_num)
        if args.with_seg is True:
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
            cb_iou = total_correct_class[6]/float(total_iou_deno_class[6])
            bolt_iou = (total_correct_class[5] + total_correct_class[6])/(float(total_iou_deno_class[5]) + float(total_iou_deno_class[6]))
            writer.add_scalar('mIoU/Train', mIoU, epoch)
            writer.add_scalar('Cbolt_IoU/Train', cb_iou, epoch)
            writer.add_scalar('Bolt_IoU/Train', bolt_iou, epoch)
            outstr = 'Train %d, Loss: %.5f, mIoU: %.5f, cb_IOU: %.5f, type cls acc: %.5f, cbolt num acc: %.5f'%(epoch,
                train_loss*1.0/count, mIoU, cb_iou, train_type_cls, train_num_acc)
        else:
            outstr = 'Train %d, Loss: %.5f, type cls acc: %.5f, cbolt num acc: %.5f'%(epoch,
                train_loss*1.0/count, train_type_cls, train_num_acc)
        io.cprint(outstr)        
        writer.add_scalar('learning rate/lr', opt.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train loss', train_loss*1.0/count, epoch)
        writer.add_scalar('Type cls/Train', train_type_cls, epoch)
        writer.add_scalar('Cbolt_Num/Train', train_num_acc, epoch)
                    
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        total_seen = 0
        total_correct = 0
        total_seen = 0
        Head.eval()
        Tail1.eval()
        test_pred_cls = []
        test_true_cls = []
        test_pred_num = []
        test_true_num = []
        total_seen_class = [0 for _ in range(num_class)]
        total_correct_class_ = [0 for _ in range(num_class)]
        total_iou_deno_class_ = [0 for _ in range(num_class)]

        for pc, seg, ty, attr, num, mask in tqdm(test_dataloader, total=len(test_dataloader), smoothing=0.9):
            pc, seg, ty, attr, num, m = pc.to(device), seg.to(device), ty.to(device), attr.to(device), num.to(device), mask.to(device)
            pc = normalize_data(pc)
            data = pc.permute(0, 2, 1)
            batch_size = data.size()[0]
            num = torch.sub(num, 3)
            pointweise, global_feature = Head(data.float())
            if args.with_seg is True:
                pred_seg, pred_ty, pred_num = Tail1(pointweise, global_feature)
                pred_seg = pred_seg.permute(0, 2, 1).contiguous().view(-1, num_class)
                loss_seg = criterion1(pred_seg, seg.view(-1, 1).squeeze())
                loss_cls = criterion1(pred_ty, ty.squeeze())               
                loss_num = criterion1(pred_num, num.squeeze())
                loss = loss_seg + loss_cls + loss_num
                pred_choice = pred_seg.cpu().data.max(1)[1].numpy()
                batch_label = seg.view(-1, 1)[:, 0].cpu().data.numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += batch_size * args.num_points
                for l in range(num_class):
                    total_seen_class[l] += np.sum(batch_label == l)
                    total_correct_class_[l] += np.sum((pred_choice == l) & (batch_label == l))     ### Intersection
                    total_iou_deno_class_[l] += np.sum((pred_choice == l) | (batch_label == l))    ### Union
            else:
                pred_ty, pred_num = Tail1(global_feature)
                loss_cls = criterion1(pred_ty, ty.squeeze())               
                loss_num = criterion1(pred_num, num.squeeze())
                loss = loss_cls + loss_num 
            logits = pred_ty.max(dim=1)[1]
            test_pred_cls.append(logits.detach().cpu().numpy())
            test_true_cls.append(ty.cpu().numpy())
            cb_num = pred_num.max(dim=1)[1]
            test_pred_num.append(cb_num.detach().cpu().numpy())
            test_true_num.append(num.cpu().numpy())           
            count += batch_size
            test_loss += loss.item() * batch_size                            
        test_pred_cls = np.concatenate(test_pred_cls)
        test_true_cls = np.concatenate(test_true_cls)
        test_type_cls = accuracy_score(test_true_cls, test_pred_cls)
        test_pred_num = np.concatenate(test_pred_num)
        test_true_num = np.concatenate(test_true_num)
        test_num_acc = accuracy_score(test_true_num, test_pred_num)
        writer.add_scalar('Loss/test loss', test_loss*1.0/count, epoch)
        writer.add_scalar('Type cls/Test', test_type_cls, epoch)
        writer.add_scalar('Cbolt_Num/Test', test_num_acc, epoch)
        if args.with_seg is True:
            test_mIoU = np.mean(np.array(total_correct_class_) / (np.array(total_iou_deno_class_, dtype=np.float64) + 1e-6))
            test_cb_iou = total_correct_class_[6] / float(total_iou_deno_class_[6])
            test_bolt_iou = (total_correct_class_[5] + total_correct_class_[6]) / (float(total_iou_deno_class_[5]) + float(total_iou_deno_class_[6]))
            writer.add_scalar('mIoU/Test', test_mIoU, epoch)
            writer.add_scalar('Cbolt_IoU/Test', test_cb_iou, epoch)
            writer.add_scalar('Bolt_IoU/Test', test_bolt_iou, epoch)
            outstr_val = 'Test %d, Loss: %.5f, mIoU: %.5f, cb_IOU: %.5f, type cls acc: %.5f, cbolt num acc: %.5f'%(epoch,
                test_loss*1.0/count, test_mIoU, test_cb_iou, test_type_cls, test_num_acc)
        else:
            outstr_val = 'Test %d, Loss: %.5f, type cls acc: %.5f, cbolt num acc: %.5f'%(epoch,
                test_loss*1.0/count, test_type_cls, test_num_acc)
        io.cprint(outstr_val) 
        
        if test_loss/count <= best_mse:
            best_mse = test_loss/count
            state1 = {'epoch': epoch, 'model_state_dict': Head.state_dict(), 'optimizer_state_dict': opt.state_dict()}
            torch.save(state1, 'outputs/%s/%s/%s/models/best_head.t7' % (args.model, args.exp_name, args.change))
            state2 = {'epoch': epoch, 'model_state_dict': Tail1.state_dict(), 'optimizer_state_dict': opt.state_dict()}
            torch.save(state2, 'outputs/%s/%s/%s/models/best_tail1.t7' % (args.model, args.exp_name, args.change))
        io.cprint('\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi Attributes Regression')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--change', type=str, default='hh', metavar='N',
                        help='explict parameters in experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--root', type=str, metavar='N',
                        help='folder of dataset')
    parser.add_argument('--csv', type=str,
                        help='motor attributes')
    parser.add_argument('--mask', type=str,
                        help='attributes mask')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    parser.add_argument('--with_seg', type=bool, default=False,
                        help='semantic segmentation in tail1 block')
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
            
        
