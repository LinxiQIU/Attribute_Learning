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
    def __init__(self, root_dir, csv_file, split='train', npoints=2048, test_area='Validation'):
        super().__init__()
        self.root_dir = root_dir
        self.npoints = npoints
        self.data = pd.read_csv(csv_file)
        motor_ls = sorted(os.listdir(root_dir))
        if split == 'train':
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) not in motor]
        else:
            motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) in motor]
        self.all_points = []
        self.all_attr = []
        self.all_type = []
        self.all_bolt_num = []
        num_points_eachmotor= []
        for idx in tqdm(motor_ids, total=len(motor_ids)):
            point_set = np.load(os.path.join(root_dir, idx))[:, 0:3]
            self.all_points.append(point_set)
            num_points_eachmotor.append(point_set.size)
            n = idx.split('_')
            attr_data = self.data[self.data['Nr.'].str.contains(n[1] + '_' + n[2])]
            attr = np.array(attr_data.iloc[:, 3:]).astype('float')
            ty = attr_data.iloc[:, 1].tolist()
            num = attr_data.iloc[:, 2].tolist()
            self.all_attr.append(attr)
            self.all_type.append(ty)
            self.all_bolt_num.append(num)
        
        
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
        
        types = self.all_type[self.motor_idxs[index]]
        
        attribute = self.all_attr[self.motor_idxs[index]]
        
        cover_bolt_num = self.all_bolt_num[self.motor_idxs[index]]
        n_points = point_set.shape[0]
        chosed = np.random.choice(n_points, self.npoints, replace=True)
        chosed_pc = point_set[chosed, :]
        sample = {'point': chosed_pc, 'attribute': torch.Tensor(attribute), 
                  'type': torch.Tensor(types), 'num': torch.Tensor(cover_bolt_num)}
        return sample


if __name__ == '__main__':
    train_data = MotorAttribute(root_dir='E:\\dataset1000', csv_file='E:\\data\\motor_attr.csv', 
                          split='test')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
    for data in train_dataloader:
        # print(data['point'].shape)
        # print(data['type'].shape)
        # print(data['attribute'].shape)
        t = data['type']
        onehot = torch.zeros(t.shape[0], 5)
        onehot.scatter_(1, t, 1.0)
        print(onehot)
            
        # bs = t.size(0)
        # one_hot = torch.nn.functional.one_hot(t, num_classes=5)
        # print(one_hot)
        