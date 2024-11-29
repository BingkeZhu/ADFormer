import numpy as np
import torch
from utils.utils import print_log
import torch.nn.functional as F
from torchvision import transforms
import kornia as K


def embedding_concat(x, y, use_cuda):
    device = torch.device('cuda:4' if use_cuda else 'cpu')
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(),align_corners=True).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection",align_corners=True)
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size(),align_corners=True).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection",align_corners=True)
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x

def norm_img(x):
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
    return x


