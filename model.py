from .layers import *
from SphericalUNetPackage.sphericalunet.utils import utils
import torch.nn as nn
import torch
from model.infoGANlayers import *

class Age_predictor(nn.Module):
    def __init__(self,in_features):
        super(Age_predictor,self).__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features,64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32,1)
        )
    def forward(self,x_l,x_r):
        x = torch.concat((x_l, x_r), dim=2)
        x = torch.mean(x,dim=2)
        return self.block(x)

'''
    ResnetEncoder:提取特征
'''
class ResNetEncoder(nn.Module):
    def __init__(self, in_c):
        super(ResNetEncoder, self).__init__()
        _,neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = utils.Get_neighs_order()

        self.conv1 = onering_conv_layer_batch(in_c, 32, neigh_orders_10242)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.LeakyReLU(0.2)

        self.pool1 = pool_layer_batch(neigh_orders_10242, 'mean')
        self.res1_1 = res_block(32, 64, neigh_orders_2562,True)
        self.res1_2 = res_block(64, 64, neigh_orders_2562)
        self.res1_3 = res_block(64, 64, neigh_orders_2562)

        self.pool2 = pool_layer_batch(neigh_orders_2562, 'mean')
        self.res2_1 = res_block(64, 128, neigh_orders_642, True)
        self.res2_2 = res_block(128, 128, neigh_orders_642)
        self.res2_3 = res_block(128, 128, neigh_orders_642)

        self.pool3 = pool_layer_batch(neigh_orders_642, 'mean')
        self.res3_1 = res_block(128, 256, neigh_orders_162, True)
        self.res3_2 = res_block(256, 256, neigh_orders_162)
        self.res3_3 = res_block(256, 256, neigh_orders_162)

        # neigh_orders = utils.Get_neighs_order(0)
        # indices = utils.Get_upconv_index(0)
        # self.up = nn.ModuleList([])
        # # self.up.append(up_block_batch_1(512, 256, indices[-2], indices[-1], neigh_orders=neigh_orders[5]))
        # self.up.append(up_block_batch_1(256, 128, indices[-4], indices[-3], neigh_orders=neigh_orders[4]))
        # self.up.append(up_block_batch_1(128, 64, indices[-6], indices[-5], neigh_orders=neigh_orders[3]))
        # self.up.append(up_block_batch_1(64, 32, indices[-8], indices[-7], neigh_orders=neigh_orders[2]))

        # self.outc = nn.Conv1d(32, out_c, kernel_size=1)

    def forward(self, x_l, x_r):
        ## left head
        x_l_0 = self.conv1(x_l)
        x_l_0 = self.bn1(x_l_0)
        x_l_0 = self.relu(x_l_0)

        x_l_1 = self.pool1(x_l_0)
        x_l_1 = self.res1_1(x_l_1)
        x_l_1 = self.res1_2(x_l_1)
        x_l_1 = self.res1_3(x_l_1)

        x_l_2 = self.pool2(x_l_1)
        x_l_2 = self.res2_1(x_l_2)
        x_l_2 = self.res2_2(x_l_2)
        x_l_2 = self.res2_3(x_l_2)

        x_l_3 = self.pool3(x_l_2)
        x_l_3 = self.res3_1(x_l_3)
        x_l_3 = self.res3_2(x_l_3)
        x_l_3 = self.res3_3(x_l_3)

        # x_l = self.pool4(x_l)
        # x_l = self.res4_1(x_l)
        # x_l = self.res4_2(x_l)
        # x_l = self.res4_3(x_l)

        # x_l = self.up[0](x_l)
        # x_l = self.up[1](x_l)
        # x_l = self.up[2](x_l)
        # # x_l = self.up[3](x_l)
        # x_l = self.outc(x_l)
        ##  right head
        x_r_0 = self.conv1(x_r)
        x_r_0 = self.bn1(x_r_0)
        x_r_0 = self.relu(x_r_0)

        x_r_1 = self.pool1(x_r_0)
        x_r_1 = self.res1_1(x_r_1)
        x_r_1 = self.res1_2(x_r_1)
        x_r_1 = self.res1_3(x_r_1)

        x_r_2 = self.pool2(x_r_1)
        x_r_2 = self.res2_1(x_r_2)
        x_r_2 = self.res2_2(x_r_2)
        x_r_2 = self.res2_3(x_r_2)

        x_r_3 = self.pool3(x_r_2)
        x_r_3 = self.res3_1(x_r_3)
        x_r_3 = self.res3_2(x_r_3)
        x_r_3 = self.res3_3(x_r_3)

        # x_r = self.pool4(x_r)
        # x_r = self.res4_1(x_r)
        # x_r = self.res4_2(x_r)
        # x_r = self.res4_3(x_r)
        #
        # x_r = self.up[0](x_r)
        # x_r = self.up[1](x_r)
        # x_r = self.up[2](x_r)
        # # x_r = self.up[3](x_r)
        # x_r = self.outc(x_r)
        return x_l_0,x_l_1,x_l_2,x_l_3,x_r_0,x_r_1,x_r_2,x_r_3

'''
    对于分割出来的两个变量进行重建，保证特异性表达是有意义的
'''

class AttentionMapGenerator(nn.Module):
    def __init__(self,in_features):
        super(AttentionMapGenerator,self).__init__()
        neigh_orders = utils.Get_neighs_order(0)
        indices = utils.Get_upconv_index(0)
        self.block = nn.Sequential(
            onering_conv_layer_batch(in_features, in_features // 2, neigh_orders=neigh_orders[5]),
            nn.BatchNorm1d(in_features // 2, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            onering_conv_layer_batch(in_features // 2, 1, neigh_orders=neigh_orders[5]),
            nn.BatchNorm1d(1, momentum=0.15, affine=True, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )
        self.up = nn.ModuleList([])
        # self.up.append(up_block_batch_1(512, 256, indices[-2], indices[-1], neigh_orders=neigh_orders[5]))
        self.up.append(up_block_batch_attention_map(in_features, 128, indices[-4], indices[-3], neigh_orders=neigh_orders[4]))
        self.up.append(up_block_batch_attention_map(128, 64, indices[-6], indices[-5], neigh_orders=neigh_orders[3]))
        self.up.append(up_block_batch_attention_map(64, 32, indices[-8], indices[-7], neigh_orders=neigh_orders[2]))

    def forward(self,x_l,x_r):

        l_map_162 = self.block(x_l)
        x_l,l_map_642 = self.up[0](x_l)
        x_l,l_map_2562 = self.up[1](x_l)
        x_l,l_map_10242 = self.up[2](x_l)
        # x_l = self.up[3](x_l)
        r_map_162 = self.block(x_r)
        x_r, r_map_642 = self.up[0](x_r)
        x_r, r_map_2562 = self.up[1](x_r)
        x_r, r_map_10242 = self.up[2](x_r)
        # x_r = self.up[3](x_r)
        return l_map_162,l_map_642,l_map_2562,l_map_10242,r_map_162,r_map_642,r_map_2562,r_map_10242

class ReconstructDecoder(nn.Module):
    def __init__(self,in_features,out_features):
        super(ReconstructDecoder,self).__init__()
        neigh_orders = utils.Get_neighs_order(0)
        indices = utils.Get_upconv_index(0)
        self.up = nn.ModuleList([])
        # self.up.append(up_block_batch_1(512, 256, indices[-2], indices[-1], neigh_orders=neigh_orders[5]))
        self.up.append(up_block_batch_1(in_features, 128, indices[-4], indices[-3], neigh_orders=neigh_orders[4]))
        self.up.append(up_block_batch_1(128, 64, indices[-6], indices[-5], neigh_orders=neigh_orders[3]))
        self.up.append(up_block_batch_1(64, 32, indices[-8], indices[-7], neigh_orders=neigh_orders[2]))
        self.outc = nn.Conv1d(32, out_features, kernel_size=1)

    def forward(self,x_l,x_r):

        x_l = self.up[0](x_l)
        x_l = self.up[1](x_l)
        x_l = self.up[2](x_l)
        # x_l = self.up[3](x_l)
        x_l = self.outc(x_l)

        x_r = self.up[0](x_r)
        x_r = self.up[1](x_r)
        x_r = self.up[2](x_r)
        # x_r = self.up[3](x_r)
        x_r = self.outc(x_r)
        return x_l,x_r

class DiseaseClassifiers(nn.Module):
    def __init__(self,in_features):
        super(DiseaseClassifiers,self).__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, x_l, x_r):
            x = torch.concat((x_l, x_r), dim=2)
            x = torch.mean(x, dim=2)
            return self.block(x)

class vertexDiseaseClassifiers(nn.Module):
    def __init__(self,in_features):
        super(vertexDiseaseClassifiers,self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_features,1,kernel_size=1),
        )
        #self.selfattention = SelfAttention(1)
        self.block_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(324, 2),
        )

    def forward(self, x_l, x_r):
            x = torch.concat((x_l, x_r), dim=2)
            x = self.block(x)
            #x = self.selfattention(x)
            return self.block_1(x)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv1d(channel, channel // ratio,  kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // ratio, channel, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self,neigh_orders):
        super(SpatialAttentionModule,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,None))
        self.max_pool = nn.AdaptiveMaxPool2d((1,None))
        self.conv = onering_conv_layer_batch(2,1,neigh_orders)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        pool = torch.cat((self.avg_pool(x),self.max_pool(x)),dim=1)
        pool_1 = self.conv(pool)
        spatialAttention = self.sigmoid(pool_1)
        return spatialAttention

class CBAM(nn.Module):
    def __init__(self, channel,neigh_orders):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule(neigh_orders)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out_age_1 = (1-self.channel_attention(x))*x
        out_age_2 = (1 - self.spatial_attention(out)) * out_age_1
        out = self.spatial_attention(out) * out
        return out,out_age_2

class simple_classifier(nn.Module):
    def __init__(self):
        super(simple_classifier,self).__init__()
        self.encoder = ResNetEncoder(3)
        self.classifier = DiseaseClassifiers(256)
        #self.generator = AttentionMapGenerator(256)


    def forward(self,x_l,x_r):
        x_l_0_ind, x_l_1_ind, x_l_2_ind, x_l_3_ind, x_r_0_ind, x_r_1_ind, x_r_2_ind, x_r_3_ind = self.encoder(
            x_l, x_r)
        #l_map_162, _, _, _, r_map_162, _, _, _ = self.generator(
            #x_l_3_ind, x_r_3_ind)
        pred_162 = self.classifier(x_l_3_ind,x_r_3_ind)
        return pred_162

class SphericalCNNEncoder(nn.Module):
    def __init__(self, input_channel):
        super(SphericalCNNEncoder,self).__init__()
        _, neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = utils.Get_neighs_order()
        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = utils.Get_upconv_index()
        self.conv1 = nn.Sequential(
            down_block_batch(input_channel, 32, neigh_orders_10242,first=True),
            down_block_batch(32, 64, neigh_orders_2562,pool_neigh_orders=neigh_orders_10242),
            down_block_batch(64, 128, neigh_orders_642,neigh_orders_2562),
            down_block_batch(128, 256, neigh_orders_162,neigh_orders_642),
        )

    def forward(self, x_l, x_r):
        x_l = self.conv1(x_l)
        x_r = self.conv1(x_r)
        return x_l,x_l,x_l,x_l,x_r,x_r,x_r,x_r

class SphericalCNN(nn.Module):
    def __init__(self):
        super(SphericalCNN,self).__init__()
        _, neigh_orders_40962, neigh_orders_10242, neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12 = utils.Get_neighs_order()
        upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242, upconv_top_index_2562, upconv_down_index_2562, upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162 = utils.Get_upconv_index()
        self.conv1 = nn.Sequential(
            down_block_batch(3, 32, neigh_orders_10242,first=True),
            down_block_batch(32, 64, neigh_orders_2562,pool_neigh_orders=neigh_orders_10242),
            down_block_batch(64, 128, neigh_orders_642,neigh_orders_2562),
            down_block_batch(128, 256, neigh_orders_162,neigh_orders_642),
            nn.Conv1d(256, 1, kernel_size=1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(324, 2),
        )

    def forward(self, x_l, x_r):
        x_l = self.conv1(x_l)
        x_r = self.conv1(x_r)
        x = torch.concat((x_l, x_r), dim=2)
        return self.fc(x)


class svgg(nn.Module):
    def __init__(self, in_ch):
        super(svgg, self).__init__()

        neigh_orders_a = utils.Get_neighs_order()
        conv_layer = onering_conv_layer_batch
        neigh_orders = neigh_orders_a[2:6]
        chs=[3,32,64,128,256]
        sequence = []
        sequence.append(conv_layer(in_ch, 32, neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(32))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(32, 32, neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(32))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))
        sequence.append(conv_layer(32, 32, neigh_orders[0]))
        sequence.append(nn.BatchNorm1d(32))
        sequence.append(nn.LeakyReLU(0.2, inplace=True))

        for l in range(1, len(chs) - 1):
            sequence.append(pool_layer_batch(neigh_orders[l-1], 'mean'))
            sequence.append(conv_layer(chs[l], chs[l + 1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l + 1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))
            sequence.append(conv_layer(chs[l + 1], chs[l + 1], neigh_orders[l]))
            sequence.append(nn.BatchNorm1d(chs[l + 1]))
            sequence.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*sequence)

    def forward(self, x_l,x_r):
        x_l = self.model(x_l)
        x_r = self.model(x_r)

        return x_l,x_l,x_l,x_l,x_r,x_r,x_r,x_r
