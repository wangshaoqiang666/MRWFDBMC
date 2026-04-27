from layer import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    # def __init__(self, num_in_node = 435, num_in_edge = 757, num_hidden1 = 512, num_out=128):  # 435, 757, 512, 128
    def __init__(self, num_in_node=89, num_in_edge=533, num_hidden1=512, num_out=64):  # 435, 757, 512, 128
        super(Model, self).__init__()
        # 第一阶段的编码器
        self.node_encoders1 = node_encoder(num_in_edge, num_hidden1, 0.3)
        self.hyperedge_encoders1 = hyperedge_encoder(num_in_node, num_hidden1, 0.3)


        self.decoder1 = decoder1(act=lambda x: x)
        self.decoder2 = decoder2(act=lambda x: x)

        # #这里应该是变分自编码器的
        self._enc_mu_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_node = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        self._enc_mu_hedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)
        self._enc_log_sigma_hyedge = node_encoder(num_hidden1, num_out, 0.3, act=lambda x: x)

        #超图卷积
        self.hgnn_node2 = HGNN1(num_in_node, num_in_node, num_out, num_in_node, num_in_node)
        self.hgnn_hyperedge2 = HGNN1(num_in_edge, num_in_edge, num_out, num_in_edge, num_in_edge)


    def sample_latent(self, z_node, z_hyperedge):
        # Return the latent normal sample z ~ N(mu, sigma^2)
        self.z_node_mean = self._enc_mu_node(z_node)  # mu
        self.z_node_log_std = self._enc_log_sigma_node(z_node)
        self.z_node_std = torch.exp(self.z_node_log_std)  # sigma
        z_node_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_node_std.size())).double()
        self.z_node_std_ = z_node_std_.cuda()
        self.z_node_ = self.z_node_mean + self.z_node_std.mul(Variable(self.z_node_std_, requires_grad=True))

        self.z_edge_mean = self._enc_mu_hedge(z_hyperedge)
        self.z_edge_log_std = self._enc_log_sigma_hyedge(z_hyperedge)
        self.z_edge_std = torch.exp(self.z_edge_log_std)
        z_edge_std_ = torch.from_numpy(np.random.normal(0, 1, size=self.z_edge_std.size())).double()
        self.z_edge_std_ = z_edge_std_.cuda()
        self.z_hyperedge_ = self.z_edge_mean + self.z_edge_std.mul(Variable(self.z_edge_std_, requires_grad=True))

        if self.training:
            return self.z_node_, self.z_hyperedge_
        else:
            return self.z_node_mean, self.z_edge_mean

    # def forward(self, AT, A , HMG, HDG, mir_feat, dis_feat, HMD, HDM , HMM , HDD):
    def forward(self, HMG, HDG, mir_feat, dis_feat,PCAassociationT,PCAassociation):


        #卷之后进行解码
        mir_feature_1 = self.hgnn_hyperedge2(mir_feat, HMG)
        dis_feature_1 = self.hgnn_node2(dis_feat, HDG)

        #变分自编码器
        z_node_encoder = self.node_encoders1(PCAassociationT)
        z_hyperedge_encoder = self.hyperedge_encoders1(PCAassociation)

        self.z_node_s, self.z_hyperedge_s = self.sample_latent(z_node_encoder, z_hyperedge_encoder)
        z_node = self.z_node_s
        z_hyperedge = self.z_hyperedge_s

        # 节点和超边特征编码
        # z_node = self.node_encoder(PCAassociationT)  # 输入形状[N_drug, num_in_node]
        # z_hyperedge = self.hyperedge_encoder(PCAassociation)  # 输入形状[N_disease, num_in_edge]

        # 后续处理与原始模型一致
        reconstructionVAE = self.decoder2(z_node, z_hyperedge)
        #
        # 变分自编码器的分布重构矩阵
        result = self.z_node_mean.mm(self.z_edge_mean.t())


        #这是超图矩阵
        reconstructionH = self.decoder1(dis_feature_1, mir_feature_1)

        # print(reconstructionH.shape[0])
        # print(reconstructionH.shape[1])
        #
        # print(PCAassociationT.shape[0])
        # print(PCAassociationT.shape[1])]

        # result_h = (reconstructionH + PCAassociationT ) / 2

        #给两个模块加权重
        a = 0.3
        result_h = a * reconstructionH + (1-a)*PCAassociationT
        # #去除主成分分析的消融
        # result_h = reconstructionH
        #这是总的矩阵
        recover = result_h

        # return   reconstructionH,  recover,  mir_feature_1, dis_feature_1
        return   reconstructionH, reconstructionVAE, result, recover,  mir_feature_1, dis_feature_1


