# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 01:31:11 2022

@author: linux
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet_utils import PointNetSetAbstraction


def knn(x, k):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        idx: sample index data, [B, N, K]
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   #(batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(1)
    num_dims = x.size(2)
    x = x.reshape(batch_size, -1, num_points)
    idx = knn(x, k)
    device = torch.device('cuda')    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx += idx_base
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature
    

def index_points_neighbors(x, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    batch_size = x.size(0)
    num_points = x.size(1)
    num_dims= x.size(2)

    device=idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx=idx+idx_base
    neighbors = x.view(batch_size*num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)

    return neighbors



def get_neighbors(x, k):
    """
    Input:
        points: input points data, [B, C, N]
    Return:
        feature_points:, indexed points data, [B, 2*C, N, K]
    """
    batch_size = x.size(0)
    num_dims= x.size(1)
    num_points = x.size(2)
    idx = knn(x, k)                                         # batch_size x num_points x 20
    x = x.transpose(2, 1).contiguous()
    neighbors = index_points_neighbors(x, idx)  
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) 
    feature = torch.cat((neighbors-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

    
class DGCNN_net(nn.Module):
    def __init__(self, output_channels=28):
        super(DGCNN_net, self).__init__()
        self.k = 32
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(64)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(512)
        self.bn9 = nn.BatchNorm1d(256)       
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(5, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(7, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1152, 512, bias=False)
        self.dp1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(256, output_channels)
        
               
    def forward(self, x, l, n):
        batch_size = x.size(0)
        
        x = get_neighbors(x, k=self.k)      # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 6*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_neighbors(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256=512, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256=512, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]
        
        l = l.view(batch_size, -1, 1)
        l = self.conv6(l)
        
        n = n.view(batch_size, -1, 1)
        n = self.conv7(n)
        
        x = torch.cat((x, l, n), dim=1)
        x = x.view(batch_size, -1)
        x = F.relu(self.bn8(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn9(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
                                 
        return x                      
        

class DGCNN_cls(nn.Module):
    def __init__(self, output_channels=5):
        super(DGCNN_cls, self).__init__()
        # self.args = args
        self.k = 32
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn8 = nn.BatchNorm1d(512)
        self.dp3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn9 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 5)
       
    def forward(self, x):
        batch_size = x.size(0)
        
        x = get_neighbors(x, k=self.k)      # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 6*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_neighbors(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256=512, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256=512, num_points) -> (batch_size, emb_dims, num_points)
        y = x.view(batch_size, -1)
        y = F.leaky_relu(self.bn8(self.fc1(y)), negative_slope=0.2)
        y = self.dp3(y)
        y = F.leaky_relu(self.bn9(self.fc2(y)), negative_slope=0.2)
        y = self.dp4(y)
        y = self.fc3(y)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x, y   # x -> type classification, y -> cover bolt numbers


class DGCNN_CORE(nn.Module):
    def __init__(self):
        super(DGCNN_CORE, self).__init__()
        self.k = 32
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)        
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4, 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5, 
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_neighbors(x, k=self.k)      # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 6*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_neighbors(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_neighbors(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_neighbors(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256=512, num_points)
        x5 = self.conv5(x)                       # (batch_size, 64+64+128+256=512, num_points) -> (batch_size, 1024, num_points)
        x = x5.max(dim=-1, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024) -> (batch_size, 1024, num_points)
        x = torch.cat((x1, x2, x3, x4, x), dim=1)   # (batch_size, 64+64+128+256+emb_dims(1024)=1536, num_points)

        return  x, x5    # x -> pointweise feature for semantic segmentation, x5 -> 1024 global feature vector



class DGCNN_Core(nn.Module):
    def __init__(self):
        super(DGCNN_Core, self).__init__()
        self.k = 32      
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),     #6*64=384
                                   self.bn1,            #2*64*2=256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),    #64*64=4096
                                   self.bn2,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #128*64=8096
                                   self.bn3,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),        #64*64=4096
                                   self.bn4,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        #0
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),      #64*64=4096
                                   self.bn5,        #256
                                   nn.LeakyReLU(negative_slope=0.2))        
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),    #192*1024=196068
                                   self.bn6,        #1024*2*2=4096
                                   nn.LeakyReLU(negative_slope=0.2))
    
    def forward(self, x):
        num_points = x.size(2)
        
        x = get_neighbors(x, k=self.k)         # (batch_size, 6, num_points) -> (batch_size, 6*2, num_points, k)
        x = self.conv1(x)                      # (batch_size, 6*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                      # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]   # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        x = get_neighbors(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_neighbors(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3), dim=1)     # (batch_size, 64+64+64=192, num_points)
        x4 = self.conv6(x)                     # (batch_size, 192, num_points) -> (batch_size, 1024, num_points)
        return x4
        # x = x4.max(dim=-1, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024)
               
        # x = x.repeat(1, 1, num_points)          # (batch_size, 1024) -> (batch_size, 1024, num_points)
        # x = torch.cat((x1, x2, x3, x), dim=1)   # (batch_size, 64+64+64+emb_dims(1024)=1216, num_points)
     
        # return x, x4  # x -> pointweise feature for semantic segmentation, x4 -> 1024 global feature vector


class TWO_CLS(nn.Module):
    def __init__(self):
        super(TWO_CLS, self).__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 5)
        
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dp3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)   # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)   # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)                              # (batch_size, emb_dims*2)
        
        ty = F.leaky_relu(self.bn1(self.linear1(x1)), negative_slope=0.2)     # (batch_size, emb_dims) -> (batch_size, 512)
        ty = self.dp1(ty)
        ty = F.leaky_relu(self.bn2(self.linear2(ty)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        ty = self.dp2(ty)
        ty = self.linear3(ty)     # (batch_size, 256) -> (batch_size, 5)
        
        num = F.leaky_relu(self.bn3(self.fc1(x)), negative_slope=0.2)      # (batch_size, 1024*2) -> (batch_size, 512)
        num = self.dp3(num)
        num = F.leaky_relu(self.bn4(self.fc2(num)), negative_slope=0.2)      # (batch_size, 512) -> (batch_size, 256)
        num = self.dp4(num)
        num = self.fc3(num)         # (batch_size, 256) -> (batch_size, 3)
        return ty, num              # ty -> type cls, num -> num of cover bolts 


class CLS_Semseg(nn.Module):
    def __init__(self):
        super(CLS_Semseg, self).__init__()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv1d(1536, 512, kernel_size=1, bias=False),         #1216*512=622592
                                   self.bn1,        #2048
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),      #512*256=131072
                                   self.bn2,    #1024
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv1d(256, 7, kernel_size=1, bias=False)   #256*6=1536
        
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 5)
        
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.dp4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp5 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 3)
        
    def forward(self, x, y):     # x (pointweise), y (1024)
        batch_size = y.size(0)
        x = self.conv1(x)           # (batch_size, 1216, num_points) -> (batch_size, 512, num_points)
        x = self.conv2(x)           # (batch_size, 512, num_points) -> (batch_size, 256, num_points)         
        x = self.dp1(x)
        x = self.conv3(x)           # (batch_size, 256, num_points) -> (batch_size, 7, num_points)
        
        y1 = F.adaptive_max_pool1d(y, 1).view(batch_size, -1)    # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)    
        y2 = F.adaptive_avg_pool1d(y, 1).view(batch_size, -1)    # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        y = torch.cat((y1, y2), 1)      # (batch_size, emb_dims*2)

        ty = F.leaky_relu(self.bn3(self.linear1(y1)), negative_slope=0.2)     # (batch_size, emb_dims) -> (batch_size, 512)
        ty = self.dp2(ty)
        ty = F.leaky_relu(self.bn4(self.linear2(ty)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        ty = self.dp3(ty)
        ty = self.linear3(ty)     # (batch_size, 256) -> (batch_size, 5)
        
        num = F.leaky_relu(self.bn5(self.fc1(y)), negative_slope=0.2)      # (batch_size, 1024*2) -> (batch_size, 512)
        num = self.dp4(num)
        num = F.leaky_relu(self.bn6(self.fc2(num)), negative_slope=0.2)      # (batch_size, 512) -> (batch_size, 256)
        num = self.dp5(num)
        num = self.fc3(num)         # (batch_size, 256) -> (batch_size, 3)
        
        return x, ty, num    # x -> semantic seg, ty -> type cls, num -> num of cover bolts 


class Attribute(nn.Module):
    def __init__(self):
        super(Attribute, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.conv1 = nn.Sequential(nn.Conv1d(5, 64, kernel_size=1, bias=False),      # (batch_size, 5)
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1152, 512)    
        # self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        # self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 28)
        
    def forward(self, x, t, n):
        batch_size = x.size(0)
        x = x.max(dim=-1, keepdim=True)[0]
        t = t.view(batch_size, -1, 1)
        t = self.conv1(t)
        n = n.view(batch_size, -1, 1)
        n = self.conv2(n)        
        x = torch.cat((x, t, n), dim=1)
        x = x.view(batch_size, -1)
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        x = self.linear3(x)     # (batch_size, 28)
        
        return x    # x -> 28 attribute
                    
    
if __name__ == '__main__':
    from torchsummary import summary
    help(summary)
    model = DGCNN_cls()
    summary(model, (3, 2048), device='cuda')
    print(model)
    