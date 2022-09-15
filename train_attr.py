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
from models import DGCNN_Core, CLS_Semseg, Attribute
from utils import cal_loss, mean_loss, IOStream, normalize_data, distance, mean_relative_error
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
    Tail1 = nn.DataParallel(CLS_Semseg().to(device))
    Tail2 = nn.DataParallel(Attribute().to(device))
    print("Let's use", torch.cuda.device_count(), "GPU!")
    params1 = list(Head.parameters()) + list(Tail1.parameters())
    params2 = list(Head.parameters()) + list(Tail2.parameters())
    if args.opt == 'sgd':
        print("Use SGD")        
        opt1 = optim.SGD(params1, lr=args.lr*100, momentum=args.momentum, 
                        weight_decay=1e-4)
        opt2 = optim.SGD(params2, lr=args.lr*100, momentum=args.momentum, 
                        weight_decay=1e-4)
    elif args.opt == 'adam':
        print("Use Adam")
        opt1 = optim.Adam(params1, lr=args.lr, weight_decay=1e-4)
        opt2 = optim.Adam(params2, lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'adamw':
        print("Use AdamW")
        opt1 = optim.AdamW(params1, lr=args.lr, weight_decay=1e-4)
        opt2 = optim.AdamW(params2, lr=args.lr, weight_decay=1e-4)
    
    if args.scheduler == 'cos':
        scheduler1 = CosineAnnealingLR(opt1, args.epochs, eta_min=1e-6)
        scheduler2 = CosineAnnealingLR(opt2, args.epochs, eta_min=1e-6)
    elif args.scheduler == 'step':
        scheduler1 = StepLR(opt1, step_size=60, gamma=0.2)
        scheduler2 = StepLR(opt2, step_size=60, gamma=0.2)
    
    print("Starting from scratch!")
    
    criterion1 = cal_loss
    criterion2 = mean_loss
    num_class = 7
    best_mse = 100
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        Head.train()
        Tail1.train()
        Tail2.train()
        train_pred_cls = []
        train_true_cls = []
        train_pred_num = []
        train_true_num = []
        train_profile_error = []
        train_gpos_xz = []
        train_gpos = []
        train_bpos = []
        train_bpos_xz = []
        train_mrot = []
        for pc, seg, ty, attr, num, mask in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9):
            pc, seg, ty, attr, num, mask = pc.to(device), seg.to(device), ty.to(device), attr.to(device), num.to(device), mask.to(device)
            pc = normalize_data(pc)
            data = pc.permute(0, 2, 1)
            batch_size = data.size()[0]
            num = torch.sub(num, 3)
            opt1.zero_grad()
            pointweise1, global_feature1 = Head(data.float())
            pred_seg, pred_ty, pred_num = Tail1(pointweise1, global_feature1)
            loss_cls = criterion1(pred_ty, ty.squeeze())
            loss_seg = criterion1(pred_seg.view(-1, num_class), seg.view(-1, 1).squeeze())
            loss_num = criterion1(pred_num, num.squeeze())
            loss1 = loss_cls + loss_seg + loss_num
            loss1.backward()
            opt1.step()
            logits = pred_ty.max(dim=1)[1]
            train_pred_cls.append(logits.detach().cpu().numpy())
            train_true_cls.append(ty.cpu().numpy())
            cb_num = pred_num.max(dim=1)[1]
            train_pred_num.append(cb_num.detach().cpu().numpy())
            train_true_num.append(num.cpu().numpy())
            sd1 = Head.state_dict()            
            Head.load_state_dict(sd1)
            opt2.zero_grad()
            pointweise2, global_feature2 = Head(data.float())                       
            type_one_hot = F.one_hot(ty.reshape(-1).long(), num_classes=5)
            num_one_hot = F.one_hot(num.reshape(-1).long(), num_classes=3)
            pred_attr = Tail2(global_feature2, type_one_hot.float(), num_one_hot.float())          
            loss2 = criterion2(pred_attr.view(-1, 28), attr.view(-1, 28), mask=mask)
            loss2.backward()
            opt2.step()
            count += batch_size
            train_loss += loss2.item() * batch_size
                        
            pred_np = pred_attr.detach().cpu().numpy()        # Size(16, 28)
            attr_np = attr.view(batch_size, -1).cpu().numpy()     # Size(16, 28)
            mask_np = mask.view(batch_size, -1).cpu().numpy()     # Size(16, 28)
            profile = np.array([x[0:4] for x in attr_np])        # Size(16, 4)
            pred_profile = np.array([x[0:4] for x in pred_np])     # Size(16, 4)
            m1 = np.array([x[0:4] for x in mask_np])                 # Size(16, 4)
            train_profile_error.append(mean_relative_error(profile, pred_profile, mask=m1)) 
            true_gpos_xz = np.array([x[i] for x in attr_np for i in [4, 6, 7, 9]])    # Size(64,)
            pred_gpos_xz = np.array([x[i] for x in pred_np for i in [4, 6, 7, 9]])
            m2 = np.array([x[i] for x in mask_np for i in [4, 7]])    #(32,) 
            train_gpos_xz.append(distance(true_gpos_xz, pred_gpos_xz, dim=2, mask=m2))
            true_gpos = np.array([x[4: 10] for x in attr_np])         #Size(16, 6)
            pred_gpos = np.array([x[4: 10] for x in pred_np])         #Size(16, 6) 
            train_gpos.append(distance(true_gpos.reshape(-1), pred_gpos.reshape(-1), dim=3, mask=m2))
            true_bpos = np.array([x[10: 25] for x in attr_np]).reshape(-1)      # Size(16*15=240)
            pred_bpos = np.array([x[10: 25] for x in pred_np]).reshape(-1)
            m3 = np.array([x[i] for x in mask_np for i in [10, 13, 16, 19, 22]])   # Size(16*5=80,)
            train_bpos.append(distance(true_bpos, pred_bpos, dim=3, mask=m3))
            bpos_xz = np.array([x[i] for x in attr_np for i in [10, 12, 13, 15, 16, 18, 19, 21, 22, 24]])      # Size(16*10=160,)
            pred_bpos_xz = np.array([x[i] for x in pred_np for i in [10, 12, 13, 15, 16, 18, 19, 21, 22, 24]])
            train_bpos_xz.append(distance(bpos_xz, pred_bpos_xz, dim=2, mask=m3))
            true_mrot = np.array([x[25: 28] for x in attr_np])     # Size(16, 3)
            pred_mrot = np.array([x[25: 28] for x in pred_np])
            train_mrot.append(np.mean(np.abs(true_mrot - pred_mrot)))
            
        if args.scheduler == 'cos':
            scheduler1.step()
            scheduler2.step()
        elif args.scheduler == 'step':
            if opt1.param_groups[0]['lr'] > 1e-5:
                scheduler1.step()
                scheduler2.step()
            if opt1.param_groups[0]['lr'] < 1e-5:
                for param_group in opt1.param_groups:
                    param_group['lr'] = 1e-5
                            
        train_pred_cls = np.concatenate(train_pred_cls)
        train_true_cls = np.concatenate(train_true_cls)
        train_type_cls = accuracy_score(train_true_cls, train_pred_cls)
        train_pred_num = np.concatenate(train_pred_num)
        train_true_num = np.concatenate(train_true_num)
        train_num_acc = accuracy_score(train_true_num, train_pred_num)
        train_profile_error = np.mean(train_profile_error)
        train_gpos_xz_error = np.mean(train_gpos_xz)
        train_gpos_error = np.mean(train_gpos)
        train_bpos_error = np.mean(train_bpos)
        train_bpos_xz_error = np.mean(train_bpos_xz)
        train_mrot_error = np.mean(train_mrot)
        outstr='Train %d, Loss: %.6f, type cls acc: %.5f, cbolts num acc: %.5f, profile error: %.5f, gear pos mdist: %.5f, gear xz mdist: %5f, cbolt mdist: %.5f, '%(epoch, train_loss*1.0/count, train_type_cls, train_num_acc, train_profile_error, train_gpos_error, train_gpos_xz_error, train_bpos_error)
        io.cprint(outstr)
        
        writer.add_scalar('learning rate/lr', opt2.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/train loss', train_loss*1.0/count, epoch)
        writer.add_scalar('Type cls/Train', train_type_cls, epoch)
        writer.add_scalar('Cbolt_Num/Train', train_num_acc, epoch)
        writer.add_scalar('Profile/Train', train_profile_error, epoch)
        writer.add_scalar('Gear_Pos/Train', train_gpos_error, epoch)
        writer.add_scalar('Gear_Pos_XZ/Train', train_gpos_xz_error, epoch)
        writer.add_scalar('Bolt_Pos/Train', train_bpos_error, epoch)
        writer.add_scalar('Bolt_Pos_XZ/Train', train_bpos_xz_error, epoch)
        writer.add_scalar('Motor_Rot/Train', train_mrot_error, epoch)
        if train_loss/count <= best_mse:
            best_mse = train_loss/count
            state1 = {'epoch': epoch, 'model_state_dict': Head.state_dict()}
            torch.save(state1, 'outputs/%s/%s/%s/models/best_head.t7' % (args.model, args.exp_name, args.change))
            state2 = {'epoch': epoch, 'model_state_dict': Tail2.state_dict()}
            torch.save(state2, 'outputs/%s/%s/%s/models/best_tail2.t7' % (args.model, args.exp_name, args.change))
            io.cprint('Best MSE at %d epoch with Loss %.6f' % (epoch, best_mse))
            
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        Head.eval()
        Tail1.eval()
        Tail2.eval()
        test_pred_cls = []
        test_true_cls = []
        test_pred_num = []
        test_true_num = []
        test_profile_error = []
        test_gpos = []
        test_gpos_xz = []
        test_bpos = []
        test_bpos_xz = []
        test_mrot = []
        for pc, seg, ty, attr, num, mask in tqdm(test_dataloader, total=len(test_dataloader), smoothing=0.9):
            pc, seg, ty, attr, num, m = pc.to(device), seg.to(device), ty.to(device), attr.to(device), num.to(device), mask.to(device)
            pc = normalize_data(pc)
            data = pc.permute(0, 2, 1)
            batch_size = data.size()[0]
            num = torch.sub(num, 3)
            pointweise, global_feature = Head(data.float())
            pred_seg, pred_ty, pred_num = Tail1(pointweise, global_feature)
            logits = pred_ty.max(dim=1)[1]
            test_pred_cls.append(logits.detach().cpu().numpy())
            test_true_cls.append(ty.cpu().numpy())
            cb_num = pred_num.max(dim=1)[1]
            test_pred_num.append(cb_num.detach().cpu().numpy())
            test_true_num.append(num.cpu().numpy())
            type_one_hot = F.one_hot(ty.reshape(-1).long(), num_classes=5)
            num_one_hot = F.one_hot(num.reshape(-1).long(), num_classes=3)
            pred_attr = Tail2(global_feature.float(), type_one_hot.float(), num_one_hot.float())
            loss = criterion2(pred_attr.view(-1, 28), attr.view(-1, 28), mask=m)
            count += batch_size
            test_loss += loss.item() * batch_size
            
            pred_np = pred_attr.detach().cpu().numpy()
            attr_np = attr.view(batch_size, -1).cpu().numpy()
            mask_np = mask.view(batch_size, -1).cpu().numpy()     # Size(16, 28)
            profile = np.array([x[0:4] for x in attr_np])        # Size(16, 4)
            pred_profile = np.array([x[0:4] for x in pred_np])     # Size(16, 4)
            m1 = np.array([x[0:4] for x in mask_np])                 # Size(16, 4)
            test_profile_error.append(mean_relative_error(profile, pred_profile, mask=m1)) 
            true_gpos_xz = np.array([x[i] for x in attr_np for i in [4, 6, 7, 9]])    # Size(64,)
            pred_gpos_xz = np.array([x[i] for x in pred_np for i in [4, 6, 7, 9]])
            m2 = np.array([x[i] for x in mask_np for i in [4, 7]])    #(32,) 
            test_gpos_xz.append(distance(true_gpos_xz, pred_gpos_xz, dim=2, mask=m2))
            true_gpos = np.array([x[4: 10] for x in attr_np])         #Size(16, 6)
            pred_gpos = np.array([x[4: 10] for x in pred_np])         #Size(16, 6) 
            test_gpos.append(distance(true_gpos.reshape(-1), pred_gpos.reshape(-1), dim=3, mask=m2))
            true_bpos = np.array([x[10: 25] for x in attr_np]).reshape(-1)      # Size(16*15=240)
            pred_bpos = np.array([x[10: 25] for x in pred_np]).reshape(-1)
            m3 = np.array([x[i] for x in mask_np for i in [10, 13, 16, 19, 22]])   # Size(16*5=80,)
            test_bpos.append(distance(true_bpos, pred_bpos, dim=3, mask=m3))
            true_bpos_xz = np.array([x[i] for x in attr_np for i in [10, 12, 13, 15, 16, 18, 19, 21, 22, 24]])      # Size(16*10=160,)
            pred_bpos_xz = np.array([x[i] for x in pred_np for i in [10, 12, 13, 15, 16, 18, 19, 21, 22, 24]])
            test_bpos_xz.append(distance(true_bpos_xz, pred_bpos_xz, dim=2, mask=m3))
            true_mrot = np.array([x[25: 28] for x in attr_np])     # Size(16, 3)
            pred_mrot = np.array([x[25: 28] for x in pred_np])
            test_mrot.append(np.mean(np.abs(true_mrot - pred_mrot)))
        test_pred_cls = np.concatenate(test_pred_cls)
        test_true_cls = np.concatenate(test_true_cls)
        test_type_cls = accuracy_score(test_true_cls, test_pred_cls)
        test_pred_num = np.concatenate(test_pred_num)
        test_true_num = np.concatenate(test_true_num)
        test_num_acc = accuracy_score(test_true_num, test_pred_num)
        test_profile_error = np.mean(test_profile_error)
        test_gpos_xz_error = np.mean(test_gpos_xz)
        test_gpos_error = np.mean(test_gpos)
        test_bpos_error = np.mean(test_bpos)
        test_bpos_xz_error = np.mean(test_bpos_xz)
        test_mrot_error = np.mean(test_mrot)
        outstr = 'Test %d, Loss: %.6f, type cls acc: %.5f, cbolts num acc: %.5f, profile error: %.5f, gear pos mdist: %.5f, gear xz midst: %.5f, cbolt mdist: %.5f'%(epoch, test_loss*1.0/count, test_type_cls, test_num_acc, test_profile_error, test_gpos_error, test_gpos_xz_error, test_bpos_error)
        io.cprint(outstr)

        writer.add_scalar('Loss/test loss', test_loss*1.0/count, epoch)
        writer.add_scalar('Type cls/Test', test_type_cls, epoch)
        writer.add_scalar('Cbolt_Num/Test', test_num_acc, epoch)
        writer.add_scalar('Profile/Test', test_profile_error, epoch)
        writer.add_scalar('Gear_Pos/Test', test_gpos_error, epoch)
        writer.add_scalar('Gear_Pos_XZ/Test', test_gpos_xz_error, epoch)
        writer.add_scalar('Bolt_Pos/Test', test_bpos_error, epoch)
        writer.add_scalar('Bolt_Pos_XZ', test_bpos_xz_error, epoch)
        writer.add_scalar('Motor_Rot/Test', test_mrot_error, epoch)

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
                        help='moter attributes')
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
            
        
