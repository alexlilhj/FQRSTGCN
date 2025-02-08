import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import config
import numpy as np

class QuantileLoss(nn.Module):
    def __init__(self, quantile_list):
        super(QuantileLoss, self).__init__()
        self.quantile_list = quantile_list

    def forward(self, preds, targets):

        error_avg = 0
        preds_quan = torch.zeros((config.quantile_list_num,targets.shape[0],2)).to(config.DEVICE)
        for i in range(config.quantile_list_num):
            for j in range(i+1):
                preds_quan[i] = torch.add(preds_quan[i],preds[j])
            preds_quan[i] = torch.exp(3*preds_quan[i])-1
        for index, quantile in enumerate(self.quantile_list):
            errors = targets - preds_quan[index]
            errors_abs = abs(errors)
            operator = torch.where(errors_abs >= 0.0005, errors_abs-0.0005/2, errors_abs*errors_abs/2/0.0005)
            losses = torch.where(errors >= 0, quantile * operator, (1 - quantile) * operator)
            error_avg += losses
        error_avg = error_avg/config.quantile_list_num
        return torch.mean(error_avg)


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):

        support = torch.matmul(input_feature, self.weight)
        output = torch.matmul(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' ('             + str(self.input_dim) + ' -> '             + str(self.output_dim) + ')'


class GcnNet(nn.Module):

    def __init__(self, input_dim=2,time_length=10,quantile_list_num=11):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 1)
        self.act1 = nn.GELU()
        self.act2 = nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d((4,time_length))
        self.lstm3 = torch.nn.LSTM(4,1,1,bidirectional=True)
        self.linear_1 = nn.Linear(time_length,quantile_list_num)

    def forward(self, adjacency, feature):

        feature = feature["flow_x"].to(config.DEVICE)
        adjacency = adjacency.to_dense()
        feature = feature.permute(0,1,3,2)
        B,N,C,T = feature.shape
        x_A = torch.Tensor(B,N,0).to(config.DEVICE)
        for t in range(T):
            x_gcn = feature[:,:,:,t]
            x_gcn = self.gcn1(adjacency,x_gcn)
            x_A = torch.cat((x_A,x_gcn),dim=2)
        feature = self.pool(x_A)
        feature = torch.transpose(feature,1,2)
        feature = torch.transpose(feature,0,1)
        feature,(hn,cn) = self.lstm3(feature)
        feature = self.act1(feature)
        feature = torch.transpose(feature,0,2)
        feature = self.linear_1(feature)
        feature = self.act2(feature)
        feature = torch.transpose(feature,0,2)

        return feature


class Gcn(nn.Module):

    def __init__(self, input_dim=2,time_length=10,quantile_list_num=11):
        super(Gcn, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 1)
        self.act1 = nn.GELU()
        self.act2 = nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d((4,time_length))
        self.linear_1 = nn.Linear(time_length,quantile_list_num)
        self.linear_2 = nn.Linear(4,2)

    def forward(self, adjacency, feature):

        feature = feature["flow_x"].to(config.DEVICE)
        adjacency = adjacency.to_dense()
        feature = feature.permute(0,1,3,2)
        B,N,C,T = feature.shape
        x_A = torch.Tensor(B,N,0).to(config.DEVICE)
        for t in range(T):
            x_gcn = feature[:,:,:,t]
            x_gcn = self.gcn1(adjacency,x_gcn)
            x_A = torch.cat((x_A,x_gcn),dim=2)
        feature = self.pool(x_A)
        feature = self.act1(self.linear_1(feature))
        feature = feature.permute(0,2,1)
        feature = self.act2(self.linear_2(feature))
        feature = feature.permute(1,0,2)

        return feature
