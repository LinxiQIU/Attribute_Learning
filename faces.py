# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 04:10:56 2022

@author: linux
"""

import pandas as pd
import numpy as np
import os
import re
import torch
import math
from torch.utils.data import Dataset
import csv
# from models import DGCNN_net
# from util import IOStream
# from torch.utils.tensorboard import SummaryWriter

def rotate_z(pos, angle):
    ro_mat =  np.array([[math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                        [0.0, 0.0, 1.0]])
    pos_new = ro_mat.dot(pos.T)
    return pos_new

def swap(ls):
    x = -1 * ls[0]
    ls[0] = ls[1]
    ls[1] = x
    return ls

def get_bolt_pos(data, idx):
    bolt_ls = data.iloc[:, 7].tolist()
    ls = re.findall(r"[+-]?\d+\.?\d*", bolt_ls[idx])
    ls = [float(x) for x in ls]
    bolt_pos = ls[12:15]
    bolt_pos.extend(ls[18:21])
    bolt_pos.extend(ls[24:27])
    if len(ls) == 42:
        bolt_pos.extend(ls[30:33])
        bolt_pos.extend(ls[36:39])
    elif len(ls) == 36:
        bolt_pos.extend(ls[30:33])
        bolt_pos.extend([0.0, 0.0, 0.0])
    elif len(ls) == 30:
        bolt_pos.extend([0.0, 0.0, 0.0])
        bolt_pos.extend([0.0, 0.0, 0.0])
    return bolt_pos


def get_ro_bolt_pos(data, idx):
    bolt_ls = data.iloc[:, 7].tolist()
    ls = re.findall(r"[+-]?\d+\.?\d*", bolt_ls[idx])
    ls = [float(x) for x in ls]
    bolt_pos = swap(ls[12:15])    
    bolt_pos.extend(swap(ls[18:21]))
    bolt_pos.extend(swap(ls[24:27]))
    if len(ls) == 42:
        bolt_pos.extend(swap(ls[30:33]))
        bolt_pos.extend(swap(ls[36:39]))
    elif len(ls) == 36:
        bolt_pos.extend(swap(ls[30:33]))
        bolt_pos.extend([0.0, 0.0, 0.0])
    elif len(ls) == 30:
        bolt_pos.extend([0.0, 0.0, 0.0])
        bolt_pos.extend([0.0, 0.0, 0.0])
    return bolt_pos

def get_dia(data, idx):
    low_dia = data.iloc[:, 3].tolist()
    # print(type(low_dia[idx]))
    # low_dia = [float(x) for x in low_dia]
    up_dia = data.iloc[:, 5].tolist()
    dia_ls = [0.0 if x == '-' else float(x) for x in up_dia ]
    # dia_mean = round(sum(dia_ls)/len(dia_ls), 3)
    # new_dia_ls = [dia_mean if x == '-' else float(x) for x in dia_ls]
    return [low_dia[idx], dia_ls[idx]]
    
def get_str_btw(s, f, b):
    par = s.partition(f)
    return (par[2].partition(b)[0][:])

def get_gear_xyz(data, idx):
    low_xyz = data.iloc[:, 4].tolist()
    low_ls = re.findall(r"[+-]?\d+\.?\d*", low_xyz[idx])
    xyz = [round(float(x), 4) for x in low_ls]
    up_xyz = data.iloc[:, 6].tolist()
    up_ls = re.findall(r"[+-]?\d+\.?\d*", up_xyz[idx])
    if len(up_ls) == 0:
        up_ls = [0.0, 0.0, 0.0]
    else:
        up_ls = [round(float(x), 3) for x in up_ls]
    xyz.extend(up_ls)
    return xyz

def get_motor_euler(data, idx):
    x = data.iloc[:, 9].tolist()
    x = round(x[idx]*180/3.14, 4)
    y = data.iloc[:, 10].tolist()
    y = round(y[idx]*180/3.14, 4)
    z = data.iloc[:, 11].tolist()
    z = round(z[idx]*180/3.14, 4)
    return [x, y, z]

def get_length(data, idx):
    bl = data.iloc[:, 1].tolist()
    sbl = data.iloc[:, 2].tolist()
    return [bl[idx], sbl[idx]]

csv_path = 'E:\\Motor_1000\\motor_parameters.csv'
root = 'E:\\dataset1000'

cols = ['Nr.', 'mf_Bottom_Length', 'mf_Sub_Bottom_Length', 'mf_Lower_Gear_ContainerDia', 
        'mf_Lower_Gear_XYZ', 'mf_Upper_Gear_ContainerDia', 'mf_Upper_Gear_XYZ', 
        'Bolts_Positions', 'Number of Bolts', 'eulerX_motor', 'eulerY_motor', 'eulerZ_motor']

df = pd.read_csv(csv_path)
data = df[cols]

dict_type = {'TypeA0': 0, 'TypeA1': 1, 'TypeA2': 2, 'TypeB0': 3, 'TypeB1': 4}
dict_bolt = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}


# m = np.loadtxt('mask_6.txt')
# print(m)

# a = pd.read_csv('attr_mask.csv')
# for idx in range(10):
#     mask = a.iloc[idx, 1:]
#     mask = np.array([mask])
#     mask = mask.astype('float')
#     print(mask)


# root_dir = 'E:\\dataset1000'
# motor_ls = sorted(os.listdir(root_dir))
# motor_ids = [motor for motor in motor_ls if '{}'.format('Validation') not in motor]

def create_csv(csv_path):
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    csv_path = csv_path + '/motor_attr_1.csv'
    with open(csv_path, 'a+', newline = '') as f:
        csv_writer = csv.writer(f)
        head = ["Nr.", "Type", "bolt_num", "BL", "sBL", "dia_1", "dia_2", "gear1_x", "gear1_y", "gear1_z",
                "gear2_x", "gear2_y", "gear2_z", "bolt1_x", "bolt1_y", "bolt1_z", "bolt2_x", "bolt2_y", "bolt2_z",
                "bolt3_x", "bolt3_y", "bolt3_z", "bolt4_x", "bolt4_y", "bolt4_z", "bolt5_x", "bolt5_y", "bolt5_z",
                "rotation_x", "rotation_y", "rotation_z"]
        csv_writer.writerow(head)



# data = pd.read_csv('E:\\data\\motor_attr.csv')
# # print(len(data))
# for idx in range(len(data)):
#     ro_data = data[data['Type'])]
        
    
# attr_ls = []
# type_ls = []
# num_ls = []
# for i, idx in enumerate(motor_ids):
# # idx = 'Validation_TypeA0_0001_cuboid'
#     # print(idx)
#     n = idx.split('_')
#     # print(n[1] + '_' + n[2])
#     attr = data[data['Nr.'].str.contains(n[1] + '_' + n[2])]
#     t = attr.iloc[:, 1].tolist()
#     num = attr.iloc[:, 2].tolist()
#     attr_data = attr.iloc[:, 3:]
#     # print(attr_data)
#     attr_data = np.array(attr_data).astype('float')
#     # print(attr_data)
#     # print(attr_data)
#     attr_ls.append(attr_data)
#     type_ls.append(t)
#     num_ls.append(num)
# print(torch.Tensor(attr_ls[0]))
# print(torch.Tensor(type_ls)[0])
# print(torch.Tensor(num_ls))

create_csv('E:\\data')
for idx in range(1000):
    name = data.iloc[:, 0].tolist()[idx]
    motor_ls = [name]
    t = dict_type[name.split('_')[0]]
    motor_ls.extend([t])
    num_bolt = data.iloc[:, 8].tolist()[idx]
    motor_ls.extend([dict_bolt[num_bolt - 2]])
    motor_ls.extend(get_length(data, idx))
    motor_ls.extend(get_dia(data, idx))
    motor_ls.extend(get_gear_xyz(data, idx))
    if (t==0) or (t==1):
        motor_ls.extend(get_bolt_pos(data, idx))
    else:
        motor_ls.extend(get_ro_bolt_pos(data, idx))
    motor_ls.extend(get_motor_euler(data, idx))

    path = 'E:\\data'
    with open(path + '\\motor_attr_1.csv', 'a+', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(motor_ls)

# for idx in range(1000):
#     name = data.iloc[:, 0].tolist()[idx]
#     mask = [name]
#     motor_ls = []
#     motor_ls.extend(get_length(data, idx))
#     motor_ls.extend(get_dia(data, idx))
#     motor_ls.extend(get_gear_xyz(data, idx))
#     motor_ls.extend(get_bolt_pos(data, idx))
#     motor_ls.extend(get_motor_euler(data, idx))
#     mask_ls = [1.0 if x != 0.0 else 0.0 for x in motor_ls]
#     mask.extend(mask_ls)
#     path = 'E:\\data'
#     with open(path + '\\attr_mask.txt', 'a+', newline='') as f:
#         for i in mask:
#             f.write('%s ' % i)
#         f.write('\n')
        # csv_writer = csv.writer(f)
        # csv_writer.writerow(mask)

class MotorAttribute(Dataset):
    def __init__(self, root_dir, csv_file, split='train', npoints=2048, test_area='Validation'):
        super().__init__()
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.npoints = npoints
        motor_ls = sorted(os.listdir(root_dir))
        if split == 'train':
            self.motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) not in motor]
        else:
            self.motor_ids = [motor for motor in motor_ls if '{}'.format(test_area) in motor]
    
        cols = ['Nr.', 'mf_Bottom_Length', 'mf_Sub_Bottom_Length', 'mf_Lower_Gear_ContainerDia', 
                'mf_Lower_Gear_XYZ', 'mf_Upper_Gear_ContainerDia', 'mf_Upper_Gear_XYZ', 
                'Bolts_Positions', 'Number of Bolts', 'eulerX_motor', 'eulerY_motor', 'eulerZ_motor']
        self.data = self.data_frame[cols]
        self.dict_type = {'TypeA0': 0, 'TypeA1': 1, 'TypeA2': 2, 'TypeB0': 3, 'TypeB1': 4}
        self.dict_bolt = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

    def __len__(self):
        return len(self.motor_ids)

    def __getitem__(self, idx):
        ### load points 
        motor_pc = np.load(os.path.join(self.root_dir,
                                        self.motor_ids[idx]))[:, 0:3]
        name = self.motor_ids[idx].split('_')
        types = self.dict_type[name[1]]
        attr = self.data[self.data['Nr.'].str.contains(name[1] + '_' + name[2])]
        attr_ls = []
        attr_ls.extend(get_length(attr, 0))
        attr_ls.extend(get_dia(attr, 0))
        attr_ls.extend(get_gear_xyz(attr, 0))
        attr_ls.extend(get_bolt_pos(attr, 0))
        attr_ls.extend(get_motor_euler(attr, 0))
        num_bolt = attr.iloc[:, 8].tolist()[0]
        num_cover_bolt = self.dict_bolt[num_bolt - 2]
        num_pc = motor_pc.shape[0]
        choose = np.random.choice(num_pc, self.npoints, replace=True)
        chosed_pc = motor_pc[choose, :]
        sample = {'point': chosed_pc, 'attribute': torch.Tensor(attr_ls), 
                  'type': torch.Tensor(types), 'num': torch.Tensor(num_cover_bolt)}
        return sample
                       

        
        
        
        
    
    
        
        

        