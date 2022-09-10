# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 01:25:31 2022

@author: linxi
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torch


class MotorAttribute(Dataset):
    def __init__(self, root_dir, csv_file, mask_file, split='train', npoints=2048, test_area='Validation'):
        super().__init__()
        self.root_dir = root_dir
        self.npoints = npoints
        self.data = pd.read_csv(csv_file)
        masks = pd.read_csv(mask_file)
        motor_ls = sorted(os.listdir(root_dir))
        if split == 'train':
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) not in motor]
        else:
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) in motor]
        self.all_points = []
        self.all_labels = []
        self.all_attr = []
        self.all_type = []
        self.all_bolt_num = []
        self.all_mask = []
        num_points_eachmotor= []
        for idx in tqdm(motor_ids, total=len(motor_ids)):
            point_data = np.load(os.path.join(root_dir, idx))
            point_set = point_data[:, 0:3]
            point_label = point_data[:, 6]
            self.all_points.append(point_set)
            self.all_labels.append(point_label)
            num_points_eachmotor.append(point_set.shape[0])
            n = idx.split('_')
            masks_data = masks[masks['Nr.'].str.contains(n[1] + '_' + n[2])]
            mask = np.array(masks_data.iloc[:, 1:]).astype('float32')
            attr_data = self.data[self.data['Nr.'].str.contains(n[1] + '_' + n[2])]
            attr = np.array(attr_data.iloc[:, 3:]).astype('float32')
            ty = np.array(attr_data.iloc[:, 1]).astype('int64')
            num = np.array(attr_data.iloc[:, 2]).astype('int64')
            self.all_attr.append(attr)
            self.all_type.append(ty)
            self.all_bolt_num.append(num)
            self.all_mask.append(mask)
        
        
        sample_prob = num_points_eachmotor / np.sum(num_points_eachmotor)
        num_inter = np.sum(num_points_eachmotor) / self.npoints
        self.motor_idxs = []
        for idx in range(len(num_points_eachmotor)):
            sample_times_onemotor = int(round(sample_prob[idx] * num_inter))
            motor_idx_onemotor = [idx] * sample_times_onemotor
            self.motor_idxs.extend(motor_idx_onemotor)
        
    def __len__(self):
        return len(self.motor_idxs)
    
    def __getitem__(self, index):
        point_set = self.all_points[self.motor_idxs[index]]
        
        labels = self.all_labels[self.motor_idxs[index]]
        
        types = self.all_type[self.motor_idxs[index]]
        
        attr = self.all_attr[self.motor_idxs[index]]
        
        cbolt_num = self.all_bolt_num[self.motor_idxs[index]]
        
        mask = self.all_mask[self.motor_idxs[index]]
        
        n_points = point_set.shape[0]
        chosed = np.random.choice(n_points, self.npoints, replace=True)
        chosed_pc = point_set[chosed, :]
        chosed_labels = labels[chosed]

        return chosed_pc, chosed_labels, types, attr, cbolt_num, mask


if __name__ == '__main__':
    train_data = MotorAttribute(root_dir='E:\\dataset1000', csv_file='E:\\data\\motor_attr.csv', 
                          mask_file='E:\\data\\attr_mask.csv', split='test')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
    for p, l, t, a, n, m in train_dataloader:
        print('type shape: ', t.shape)
        print('num shape: ', n.shape)
        print(m.shape)
        
        
        
        