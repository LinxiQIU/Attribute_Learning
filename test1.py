# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 21:29:33 2022

@author: linux
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data_attr import MotorAttribute
# from utils import distance


def mean_loss(pred, target, mask=None):
    mse_loss = torch.nn.MSELoss(reduction='none')
    loss = mse_loss(pred, target)        #（16，28）
    # loss = torch.sum(loss, dim=1)        #（16，1）
    if mask is not None:
        loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def mean_relative_error(y_true, y_pred, mask):
    error = []
    for i in range(len(y_true)):
        e = np.abs(y_true[i] - y_pred[i])
        if mask[i][3] == 0.0:
            er = e[:3] / y_true[i][:3]
        else:
            er = e / y_true[i]
        error.append(np.sum(er)/np.sum(mask[i]))
    return np.mean(error)


def distance(ls1, ls2, dim, mask=None):
    if dim == 2:
        l1 = np.array_split(ls1, len(ls1)/2)
        l2 = np.array_split(ls2, len(ls2)/2)
        # m = np.array_split(mask, len(mask)/2)
    elif dim == 3:
        l1 = np.array_split(ls1, len(ls1)/3)
        l2 = np.array_split(ls2, len(ls2)/3)
        # m = np.array_split(mask, len(mask)/2)
    dist = []
    for i in range(len(l1)):
        squared_dist = np.sum((l1[i] - l2[i])**2, axis=0)
        d = np.sqrt(squared_dist)
        dist.append(d)        
    valid_dist = np.sum(dist * mask) / np.sum(mask)
    return valid_dist

train_data = MotorAttribute(root_dir='E:\\dataset1000', csv_file='E:\\data\\motor_attr.csv', 
                      mask_file='E:\\data\\attr_mask.csv', split='test')
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)


train_bpos = []
for p, l, t, A, n, m in tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9):
# for p, l, t, A, n, m in train_dataloader:
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
    
    
    attr_np = A.view(16, -1).numpy()     # Size(16, 28)
    m_np = m.view(16, -1).numpy()       # Size(16, 28)
    pred_np = pred.detach().numpy()     # Size(16, 28)
    # true_mrot = np.array([x[25: 28] for x in attr_np])     # Size(16, 3)
    # pred_mrot = np.array([x[25: 28] for x in pred_np])
    # mrot = np.abs(true_mrot - pred_mrot)
    # print(mrot)
    # train_mrot = np.mean(mrot)
    # print(train_mrot)
    
    
    num = torch.sub(n, 3)
    print(num.shape)
    # print(n.shape)
    # profile = np.array([x[0:4] for x in attr_np])   # Size(16, 4)
    # # print(profile)
    # pred_profile = np.array([x[0:4] for x in pred_np])   # Size(16, 4)
    # # print(pred_profile)
    # m1 = np.array([x[0:4] for x in m_np])   # Size(16, 4)
    # # print(m1)
    # error = mean_relative_error(profile, pred_profile, m1)
    # print(error)
    
    # true_gpos_xz = np.array([x[i] for x in attr_np for i in [4, 6, 7, 9]])    # Size(64,)
    # pred_gpos_xz = np.array([x[i] for x in pred_np for i in [4, 6, 7, 9]])
    # m2 = np.array([x[i] for x in m_np for i in [4, 7]])    #(32,)
    # print(len(m2))
    # xz_dist = distance(true_gpos_xz, pred_gpos_xz, dim=2, mask=m2)
    # print(xz_dist)
    
    # print(gpos.shape)  # 64
    # train_true_gpos_xz.append(np.array([x[i] for x in target_np for i in [4, 6, 7, 9]]))
    # train_pred_gpos_xz.append(np.array([x[i] for x in pred_np for i in [4, 6, 7, 9]]))
    # print()
    # dist_xz = 
    # print(gpos_xz)
    # pred_profile = np.array([x[0:4] for x in pred_np])   # Size(16, 4)    
    # m_np = m.numpy().reshape(16, -1)     # Size(16, 28)
    # print(m_np.shape)
        
    # print(np.mean(error2))
    # m1 = m[:4]
    # mean_relative_error(profile, y_pred)
    # pred_gpos = np.array([x[4:10] for x in pred_np])  #Size(16, 6)
    # gpos = np.array([x[4:10] for x in attr_np])
    # # print(gpos)
    # # print(gpos.reshape(-1))
    # dist = distance(pred_gpos.reshape(-1), gpos.reshape(-1), dim=3, mask=m2)
    # print(dist)
    bpos_xz = np.array([x[i] for x in attr_np for i in [10, 12, 13, 15, 16, 18, 19, 21, 22, 24]])      # Size(16, 15)
    print(bpos_xz)
    print(bpos_xz.shape)
    pred_bpos_xz = np.array([x[i] for x in pred_np for i in [10, 12, 13, 15, 16, 18, 19, 21, 22, 24]])
    print(pred_bpos_xz.shape)
    
    m3 = np.array([x[i] for x in m_np for i in [10, 13, 16, 19, 22]])
    # # print(m3.shape)    # Size(16*5=80,)
    dist3 = distance(bpos_xz, pred_bpos_xz, dim=2, mask=m3)
    print(dist3)
    # train_bpos.append(dist3)
#     print(len(train_bpos))
# train_bpos_dist_mean = np.mean(train_bpos)
# print('Bolt_Pos in one epoch: ', train_bpos_dist_mean)
    
    # train_true_cls = np.concatenate(profile.reshape(-1))
    # train_pred_cls = np.concatenate(pred_profile.reshape(-1))
    # train_true_cls.append(profile.reshape(-1))
    # print(len(train_true_cls))
    # train_pred_cls.append(pred_profile.reshape(-1))
    # print(len(train_pred_cls))
    # print(mean_relative_error(train_true_cls, train_pred_cls))
# train_true_cls = np.concatenate(train_true_cls)
# print(train_true_cls.shape)
# train_pred_cls = np.concatenate(train_pred_cls)
# print(train_pred_cls.shape)
# error = mean_relative_error(train_true_cls, train_pred_cls)
# print(error)
    
    # gear_pos = 
    # print(pred_body.shape)
    # print(mape(train_true_cls, train_pred_cls))
    
    # Loss = mean_loss(pred.view(-1, 28), target.view(-1, 28), mask=m)
    # print(m.shape)
    # Loss.backward()
    # print(Loss)

