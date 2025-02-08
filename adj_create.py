import scipy.sparse as sp
import numpy as np
import config
import pandas as pd
import torch
import csv

def normalize_data(max_data, min_data, data):
    mid = min_data
    base = max_data - min_data
    normalized_data = (data - mid) / base

    return normalized_data

def normalization(adjacency):
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


def create_adj_all():
    A = np.zeros([32, 32])
    with open("relationship_all.csv", "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])
            A[i, j] = 1. / distance
            A[j, i] = 1. / distance
        matrix_i = torch.eye(32, dtype=torch.float, device=config.DEVICE)
        B = torch.tensor(A).to(config.DEVICE) + matrix_i
        degree_matrix = torch.sum(B, dim=1, keepdim=False)
        degree_matrix = degree_matrix.pow(-0.5)
        degree_matrix = torch.diag(degree_matrix)

    return torch.mm(degree_matrix, torch.mm(B,degree_matrix))



def create_adj_global():
    edge_owner_global = pd.read_excel(r'relationship_all.xlsx')
    edge_owner_global = np.array(edge_owner_global)
    adj_weight = np.ones(edge_owner_global.shape[0])
    for i in range(edge_owner_global.shape[0]):
        fushu = complex(edge_owner_global[i][2])
        adj_weight[i] = 1/abs(fushu)
    adj_global = sp.coo_matrix((adj_weight, (edge_owner_global[:, 0], edge_owner_global[:, 1])),
                               shape=(32,32),
                               dtype=np.float32)
    adj_global = adj_global + adj_global.T.multiply(adj_global.T > adj_global) - adj_global.multiply(adj_global.T > adj_global)

    normalize_adj_global =normalization(adj_global)
    indices = torch.from_numpy(np.asarray([normalize_adj_global.row,
                                           normalize_adj_global.col]).astype('int64')).long()
    values = torch.from_numpy(normalize_adj_global.data.astype(np.float32))
    tensor_adj_global=torch.sparse.FloatTensor(indices,values,(32, 32)).to(config.DEVICE)
    return tensor_adj_global


def create_adj_owneri(owner_i):
    a = np.array([16,4,11])
    edge_owner = pd.read_excel(r'relationship_'+str(owner_i)+'.xlsx')
    edge_owner = np.array(edge_owner)
    adj_weight_owner = np.ones(edge_owner.shape[0])
    for i in range(edge_owner.shape[0]):
        fushu_owner = complex(edge_owner[i][2])
        adj_weight_owner[i] = 1/abs(fushu_owner)
    adj_owner = sp.coo_matrix((adj_weight_owner, (edge_owner[:, 0], edge_owner[:, 1])),
                              shape=(a[owner_i], a[owner_i]),
                              dtype=np.float32)
    adj_owner = adj_owner + adj_owner.T.multiply(adj_owner.T > adj_owner) - adj_owner.multiply(adj_owner.T > adj_owner)

    normalize_adj = normalization(adj_owner)
    indices = torch.from_numpy(np.asarray([normalize_adj.row,
                                           normalize_adj.col]).astype('int64')).long()
    values = torch.from_numpy(normalize_adj.data.astype(np.float32))
    tensor_adj = torch.sparse.FloatTensor(indices, values,
                                          (a[owner_i], a[owner_i])).to(config.DEVICE)
    return tensor_adj

