import torch
from net import GcnNet,QuantileLoss,Gcn,LSTMnet,CNNnet,MLPnet,Transformernet
import torch.optim as optim
import numpy as np
import torch.nn as nn
import config
from phe import paillier



class Aggregate:
    def __init__(self,owner_i,train_data_loader_owneri,adj_owneri,input_length,quantile_list_num):
        self.train_data_loader = train_data_loader_owneri
        self.adj_owneri = adj_owneri
        self.INPUT_DIM = config.INPUT_DIM
        self.DEVICE = config.DEVICE
        self.LEARNING_RATE = config.LEARNING_RATE
        self.local_EPOCHS = config.local_epoch
        self.input_length = input_length
        self.quantile_list_num = quantile_list_num
        self.localsage = self.build_aggregator()
        self.number_list = owner_i
        self.input_length = input_length
        self.quantile_list_num = quantile_list_num

    def build_aggregator(self):
        model = GcnNet(input_dim=self.INPUT_DIM,time_length = self.input_length,quantile_list_num = self.quantile_list_num).to(self.DEVICE)
        return model

    def local_train(self,global_model,number_list,quantile_list):

        for name, param in global_model.state_dict().items():
            self.localsage.state_dict()[name].copy_(param.clone())

        optimizer = optim.Adam(self.localsage.parameters(), lr=self.LEARNING_RATE, weight_decay=0.000001)

        loss_function = QuantileLoss(quantile_list)
        self.localsage.train()
        for i in range(self.local_EPOCHS):
            loss_epoch_fed = 0
            for data in self.train_data_loader:

                logits = self.localsage(self.adj_owneri,data)
                true_label = data["flow_y"].to(config.DEVICE)
                loss = loss_function(logits, true_label.to(torch.int64))
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                loss_epoch_fed += loss.item()
            print("Fed local train: owner: {:03d} local_Epoch: {:03d} Train_Loss: {:.4f}".format(number_list,i,loss_epoch_fed))
        diff = dict()
        for name, data in self.localsage.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])

        return diff
