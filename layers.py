import numpy as np
import torch.nn as nn
import SphericalUNetPackage.sphericalunet.utils.utils as utils
import torch
import scipy.io as sio
import os
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
abspath="/Users/xuepengcheng/Xue/CHD_classification/SphericalUNetPackage/sphericalunet/utils"
class onering_conv_layer(nn.Module):
    def __init__(self, in_features, out_features, neigh_orders):
        super(onering_conv_layer,self).__init__()

        self.in_features = in_features
        self.out_featrues = out_features
        self.neigh_orders = neigh_orders
        self.weight = nn.Linear(7*in_features,out_features)

    def forward(self,x):
        mat = x[self.neigh_orders]
        mat = mat.view(len(x),7*self.in_features)
        out_features = self.weight(mat)
        return out_features

class onering_conv_layer_batch(nn.Module):
    def __init__(self, in_features,out_features,neigh_orders):
        super(onering_conv_layer_batch,self).__init__()

        self.in_features = in_features
        self.out_featrues = out_features
        self.neigh_orders = neigh_orders
        self.weight = nn.Linear(7 * in_features, out_features)
    ## x.shape = N * features * vertices
    def forward(self,x):
        mat = x[:,:, self.neigh_orders]
        mat = mat.view(x.shape[0], self.in_features, x.shape[2],7).permute(0,2,3,1)
        mat = mat.contiguous().view(x.shape[0],x.shape[2],7*self.in_features)
        out_features = self.weight(mat).permute(0,2,1)
        return out_features

class pool_layer(nn.Module):
    def __init__(self,neigh_orders,pool_type="mean"):
        super().__init__()
        self.neigh_orders = neigh_orders
        self.pool_type = pool_type
    # x.shape = N * output_features
    def forward(self,x):
        number_nodes = int((x.size()[0]+6)/4)
        features_num = x.size()[1]
        x = x[self.neigh_orders[0:number_nodes*7]].view(number_nodes,7,features_num)
        if self.pool_type == "mean":
            x=torch.mean(x,dim=1)
        if self.pool_type == "max":
            x = torch.max(x,dim=1)
            return x[0],x[1]
        return x
class pool_layer_batch(nn.Module):
    def __init__(self,neigh_orders,pool_type="mean"):
        super().__init__()
        self.neigh_orders=neigh_orders
        self.pool_type=pool_type
    # x.shape = B * output_features * N
    def forward(self,x):
        number_nodes = int((x.size()[2]+6)/4)
        features_number = x.size()[1]
        x = x[:,:,self.neigh_orders[0:number_nodes*7]]
        x = x.view(x.size()[0],features_number,number_nodes,7)
        if self.pool_type == "mean":
            x = torch.mean(x, dim=3)
        if self.pool_type == "max":
            x = torch.max(x, dim=3)
            return x[0]
        return x

class upconv_layer(nn.Module):
    def __init__(self, in_features, out_features, upconv_center_indices, upconv_edge_indices):
        super(upconv_layer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.upcon_center_indices = upconv_center_indices
        self.upcon_edge_indices = upconv_edge_indices
        self.weight = nn.Linear(in_features,7*out_features)
    # N*in_features
    def forward(self,x):
        raw_nodes = x.size()[0]
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x)
        x = x.view(x.shape[0]*7, self.out_features)
        x1 = x[self.upcon_center_indices]
        assert (x1.size() == torch.Size([raw_nodes, self.out_features]))
        x2 = x[self.upcon_edge_indices].view(-1,self.out_features,2)
        x = torch.cat((x1,torch.mean(x2,dim=2)),0)
        assert(x.size() == torch.Size([new_nodes, self.out_features]))
        return x

class upconv_layer_batch(nn.Module):
    def __init__(self, in_features,out_features,upconv_center_indices,upconv_edge_indices):
        super(upconv_layer_batch,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.upconv_center_indices = upconv_center_indices
        self.upconv_edge_indices = upconv_edge_indices
        self.weight = nn.Conv1d(in_features, 7 * out_features, kernel_size=1)
    # input N * vertices * features
    def forward(self,x):
        raw_nodes = x.size()[2]
        new_nodes = int(raw_nodes * 4 - 6)
        x = self.weight(x) # N * (7*out_features) * vertices
        x = x.permute(0,2,1)
        x = x.contiguous().view(x.shape[0],raw_nodes*7,self.out_features).permute(0,2,1)

        x1 = x[:, :, self.upconv_center_indices]
        assert (x1.size() == torch.Size([x.shape[0], self.out_features, raw_nodes]))
        x2 = x[:, :, self.upconv_edge_indices].view(x.shape[0], self.out_features, -1, 2)
        x = torch.cat((x1, torch.mean(x2, 3)), 2)
        assert (x.size() == torch.Size([x.shape[0], self.out_features, new_nodes]))
        # x = self.norm(x)
        return x


class down_block(nn.Module):
    def __init__(self, conv_layer, in_ch, out_ch, neigh_orders, pool_neigh_orders, first=False):
        super(down_block,self).__init__()
        # no pooling
        if first:
            self.block = nn.Sequential(
                conv_layer(in_ch,out_ch,neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15,affine=True,track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch,out_ch,neigh_orders),
                nn.BatchNorm1d(out_ch,momentum=0.15,affine=True,track_running_stats=False),
                nn.LeakyReLU(0.2,inplace=True)
            )
        else:
            self.block = nn.Sequential(
                pool_layer(pool_neigh_orders,"mean"),
                conv_layer(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                conv_layer(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self,x):
        out_features= self.block(x)
        return out_features
class down_block_batch(nn.Module):
    def __init__(self, in_ch, out_ch, neigh_orders, pool_neigh_orders = None,first=False):
        super(down_block_batch,self).__init__()
        if first:
            self.block = nn.Sequential(
                onering_conv_layer_batch(in_ch,out_ch,neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer_batch(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                pool_layer_batch(pool_neigh_orders,"mean"),
                onering_conv_layer_batch(in_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True),
                onering_conv_layer_batch(out_ch, out_ch, neigh_orders),
                nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self,x):
        out_features= self.block(x)
        return out_features

class hierarchical_down_block_batch(nn.Module):
    def __init__(self, in_ch, out_ch, neigh_orders, pool_neigh_orders = None):
        super(hierarchical_down_block_batch,self).__init__()

        self.block = nn.Sequential(
            pool_layer_batch(pool_neigh_orders,"mean"),
            onering_conv_layer_batch(in_ch, out_ch, neigh_orders),
            nn.BatchNorm1d(out_ch, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.weight = nn.Conv1d(out_ch*2,out_ch,kernel_size=1)
    def forward(self, x,x1):
        out_features = self.block(x)
        out_features = torch.cat((out_features,x1),dim=1)
        out_features = self.weight(out_features)
        return out_features
class up_block(nn.Module):
    def __init__(self,in_features,out_features,upconv_center_indices,upconv_edge_indices,neigh_orders):
        super(up_block,self).__init__()
        self.up = upconv_layer(in_features,out_features,upconv_center_indices,upconv_edge_indices)
        self.block = nn.Sequential(
            onering_conv_layer(out_features*2,out_features,neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            onering_conv_layer(out_features, out_features, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self,x1,x2):
        up = self.up(x1)
        x = torch.cat((up,x2),1)
        x = self.block(x)
        return x

class up_block_batch(nn.Module):
    def __init__(self,in_features,out_features,upconv_center_indices,upconv_edge_indices,neigh_orders):
        super(up_block_batch,self).__init__()
        self.up = upconv_layer_batch(in_features, out_features, upconv_center_indices, upconv_edge_indices)
        self.block = nn.Sequential(
            onering_conv_layer_batch(out_features * 2, out_features, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            onering_conv_layer_batch(out_features, out_features, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self,x1,x2):
        up = self.up(x1)
        x = torch.cat((up,x2),1)
        x = self.block(x)
        return x

class up_block_batch_1(nn.Module):
    def __init__(self,in_features,out_features,upconv_center_indices,upconv_edge_indices,neigh_orders):
        super(up_block_batch_1,self).__init__()
        self.up = upconv_layer_batch(in_features, out_features, upconv_center_indices, upconv_edge_indices)
        self.block = nn.Sequential(
            onering_conv_layer_batch(out_features, out_features, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self,x1):
        up = self.up(x1)
        up = self.block(up)
        return up

class up_block_batch_attention_map(nn.Module):
    def __init__(self,in_features,out_features,upconv_center_indices,upconv_edge_indices,neigh_orders):
        super(up_block_batch_attention_map,self).__init__()
        self.up = nn.Sequential(
            upconv_layer_batch(in_features, out_features, upconv_center_indices, upconv_edge_indices),
            onering_conv_layer_batch(out_features, out_features, neigh_orders=neigh_orders),
        )
        self.block = nn.Sequential(
            onering_conv_layer_batch(out_features, out_features//2, neigh_orders=neigh_orders),
            nn.BatchNorm1d(out_features // 2, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            onering_conv_layer_batch(out_features//2, 1, neigh_orders=neigh_orders),
            nn.Sigmoid()
        )
    def forward(self,x1):
        up = self.up(x1)
        map = self.block(up)
        return up,map


class res_block(nn.Module):
    def __init__(self, c_in, c_out, neigh_orders, first_in_block=False):
        super(res_block, self).__init__()

        self.conv1 = onering_conv_layer_batch(c_in, c_out, neigh_orders)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = onering_conv_layer_batch(c_out, c_out, neigh_orders)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.first = first_in_block


    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.first:
            res = torch.cat((res, res), 1)
        x = x + res
        x = self.relu(x)

        return x




