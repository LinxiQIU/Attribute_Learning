
import numpy as np
import torch
import torch.nn.functional as F
import random


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    gold = gold.type(torch.int64)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def mean_loss(pred, target, mask=None):
    mse_loss = torch.nn.MSELoss(reduction='none')
    loss = mse_loss(pred, target)          #（batch_size, 28）
    # loss = torch.mean(loss, dim=1)
    if mask is not None:
        loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


class PrintLog():
    def __init__(self, path):
        self.f = open(path, 'a')        # 'a' is used to add some contents at end  of current file

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()      #to ensure the line will be wroten and the content in buffer will get deleted

    def close(self):
        self.f.close()


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    for b in range(B):
        pc = batch_data[b]
        centroid = torch.mean(pc, dim=0,keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1,keepdim=True)))
        pc = pc / m
        batch_data[b] = pc
    return batch_data


def rotate_180_z(data):
    """ 
        Input:
          BXNx6 array, original batch of point clouds and point normals
        Return:
          BXNx3 array, rotated batch of point clouds
    """
    data=data.float()
    rotated_data = torch.zeros(data.shape, dtype=torch.float32)
    rotated_data = rotated_data.cuda()
    angles=[0,0,np.pi]
    angles=np.array(angles)
    for k in range(data.shape[0]):
        
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        R=torch.from_numpy(R).float().cuda()
        rotated_data[k,:,:] = torch.matmul(data[k,:,:], R)
    return rotated_data


def rotate(data,angle_clip=np.pi*0.25):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BXNx6 array, original batch of point clouds and point normals
        Return:
          BXNx3 array, rotated batch of point clouds
    """
    data=data.float()
    rotated_data = torch.zeros(data.shape, dtype=torch.float32)
    rotated_data = rotated_data.cuda()
    angles=[]
    batch_size=data.shape[0]
    rotation_matrix=torch.zeros((batch_size,3,3),dtype=torch.float32).cuda()
    for i in range(3): 
        angles.append(random.uniform(-angle_clip,angle_clip))
    angles=np.array(angles)
    for k in range(data.shape[0]):
        
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        R=torch.from_numpy(R).float().cuda()
        rotated_data[k,:,:] = torch.matmul(data[k,:,:], R)
        rotation_matrix[k,:,:]=R
    return rotated_data,rotation_matrix




def rotate_per_batch(data,angle_clip=np.pi*1):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BXNx6 array, original batch of point clouds and point normals
        Return:
          BXNx3 array, rotated batch of point clouds
    """
    data=data.float()
    rotated_data = torch.zeros(data.shape, dtype=torch.float32)
    rotated_data = rotated_data.cuda()
    batch_size=data.shape[0]
    rotation_matrix=torch.zeros((batch_size,3,3),dtype=torch.float32).cuda()
    for k in range(data.shape[0]):
        angles=[]
        for i in range(3): 
            angles.append(random.uniform(-angle_clip,angle_clip))
        angles=np.array(angles)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        R=torch.from_numpy(R).float().cuda()
        rotated_data[k,:,:] = torch.matmul(data[k,:,:], R)
        rotation_matrix[k,:,:]=R
    return rotated_data,rotation_matrix


def feature_transform_reguliarzer(trans,GT=None):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    if GT ==None:
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    else:
        loss = torch.mean(torch.norm(trans - GT, dim=(1, 2)))
    return loss


def get_parameter_number(net):
    total=0
    times=0
    for p in net.parameters():
        inter=p.numel()
        times=times+1
        total=total+inter
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


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


def mean_relative_error(y_true, y_pred, mask):
    error = []
    for i in range(len(y_true)):
        e = np.abs(y_true[i] - y_pred[i])
        if mask[i][3] < 1e-9:
            er = e[:3] / y_true[i][:3]
        else:
            er = e / y_true[i]
        error.append(np.sum(er)/np.sum(mask[i]))
    return np.mean(error)