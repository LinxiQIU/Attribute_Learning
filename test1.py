# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 21:29:33 2022

@author: linux
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_attr import MotorAttribute
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
# from sklearn.metrics import 

def mean_loss(pred, target, mask=None):
    mse_loss = torch.nn.MSELoss(reduction='none')
    loss = mse_loss(pred, target)        #（16，28）
    # loss = torch.sum(loss, dim=1)        #（16，1）
    if mask is not None:
       
        # print(mask1.shape)
    #     # loss = loss*mask
    #     # loss1 = torch.mean(loss * mask)
        loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


train_data = MotorAttribute(root_dir='E:\\dataset1000', csv_file='E:\\data\\motor_attr.csv', 
                      mask_file='E:\\data\\attr_mask.csv', split='test')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)


train_true_cls = []
train_pred_cls = []
for p, l, t, A, n, m in train_dataloader:
    x = torch.randn(16, 7)
    # print(x.shape)
    a = x.view(16, -1, 1)
    # print(a.shape)
    
    conv1 = nn.Sequential(nn.Conv1d(7, 64, kernel_size=1, bias=False), 
                          nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))
    # print(conv1(a).shape)
    
    y = torch.randn(16, 5)
    b = y.view(16, -1, 1)
    conv2 = nn.Sequential(nn.Conv1d(5, 64, kernel_size=1, bias=False), 
                          nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))
    # print(conv2(b).shape)
    c = torch.randn(16, 1024, 2048)
    # print(c.shape)
    c = c.max(dim=-1, keepdim=True)[0]
    # print(c.shape)
    z = torch.cat((c, conv1(a), conv2(b)), dim=1)
    # print(z.shape)
    l1 = nn.Linear(1152, 512)
    dp1 = nn.Dropout(p=0.5)
    bn1 = nn.BatchNorm1d(512)
    e = F.leaky_relu(bn1(l1(z.view(16, -1))))
    l2 = nn.Linear(512, 28)
    pred = l2(e)
    
    target = torch.randn(16, 28)
    pred_np = pred.detach().numpy()
    target_np = target.numpy()
    # profile = np.array([x[:4] for x in target_np])    # (16, 4)   
    # idxs = [4,6,7,9]
    pred_profile = np.array([x[i] for x in pred_np for i in [4,6,7,9]])
    pred_profile1 = np.array([x[4:10] for x in pred_np])
    print(pred_profile)
    print(pred_profile1)
    print(pred_profile.shape)
    # print(pred_profile.reshape(-1).shape)
    # train_true_cls.append(np.array([x[] for x in target_np]).reshape(-1))
    # train_pred_cls.append(pred_profile.reshape(-1))
    # gear_pos = 
    # print(pred_body.shape)
    # print(mape(train_true_cls, train_pred_cls))
    
    Loss = mean_loss(pred.view(-1, 28), target.view(-1, 28), mask=m)
    # print(m.shape)
    # Loss.backward()
    # print(Loss)

