import torch
import config
import numpy as np
import torch.nn as nn
from net import GcnNet,QuantileLoss,Gcn,LSTMnet,CNNnet,MLPnet,Transformernet
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from ada import ada
from phe import paillier

def train_fedSage(aggregator_list: list, num_owners,testdata_loader,communication_num,tensor_adj_global,test_owner_all_loader,quantile_list,global_pub_key, global_priv_key):
    print('fed_learning begin')
    global_model = GcnNet(input_dim=config.INPUT_DIM).to(config.DEVICE)

    logger = SummaryWriter(log_dir="runs_path", flush_secs=1)
    local_loss_quantile = []
    global_loss_quantile = []
    for ec in range(communication_num):
        weight_accumulator = {}
        for name, params in global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)


        print("local training begin: Iterarion: {:03d}".format(ec))
        for aggregator in aggregator_list:

            # ALA_a = ALA(QuantileLoss(quantile_list), aggregator.train_data_loader, aggregator.adj_owneri,config.batch_size,  1, config.DEVICE)
            # ALA_a.adaptive_local_aggregation(global_model,aggregator.localsage)
            # diff = aggregator.local_train(aggregator.localsage,aggregator.number_list,quantile_list)


            diff = aggregator.local_train(global_model,aggregator.number_list,quantile_list)
            for name, params in global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        for k in weight_accumulator.keys():
            list_w = weight_accumulator[k].view(-1).tolist()
            for i, elem in enumerate(list_w):
                list_w[i] = global_pub_key.encrypt(elem)
            weight_accumulator[k] = list_w


        for name, data in global_model.state_dict().items():
            for i in range(len(weight_accumulator[name])):
                weight_accumulator[name][i] /= num_owners
            update_per_layer = weight_accumulator[name]
            for i,elem in enumerate(update_per_layer):
                update_per_layer[i] = global_priv_key.decrypt(elem)
            original_shape = list(global_model.state_dict()[name].size())
            update_per_layer = torch.FloatTensor(update_per_layer).to(config.DEVICE).view(*original_shape)

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)


        with torch.no_grad():
            loss_test_epoch_all = 0
            for i in range(config.num_owners):
                Predict_local = np.zeros([config.quantile_list_num,1,2])
                Target_local = np.zeros([1,2])
                loss_test_epoch_local = 0
                local_model = aggregator_list[i].localsage
                local_model.eval()
                adj_owneri = aggregator_list[i].adj_owneri
                test_data_owneri = testdata_loader[i]
                for data in test_data_owneri:
                    loss_function = QuantileLoss(quantile_list)
                    test_logits = local_model(adj_owneri,data)
                    test_label = data["flow_y"].to(config.DEVICE)
                    loss = loss_function(test_logits,test_label.to(torch.int64))
                    loss_test_epoch_local += loss.item()
                    loss_test_epoch_all += loss.item()
                    m = torch.zeros((config.quantile_list_num,test_label.shape[0],2)).to(config.DEVICE)
                    for a in range(config.quantile_list_num):
                        for b in range(a+1):
                            m[a] = torch.add(m[a],test_logits[b])
                        m[a] = torch.exp(3*m[a])-1
                    Target_local = np.concatenate([Target_local,np.array(test_label.cpu())],axis=0)
                    Predict_local = np.concatenate([Predict_local,np.array(m.cpu())],axis=1)
                Target_local_pd = pd.DataFrame(Target_local)
                for j in range(config.quantile_list_num):
                    Predict_local_pd = pd.DataFrame(Predict_local[j])
                    filename_local_prediction = r'F:\local_predict_quantile_'+str(j)+r'_owner'+str(i)+r'.xlsx'
                    with pd.ExcelWriter(filename_local_prediction) as writer:
                        Target_local_pd.to_excel(writer,sheet_name="target",float_format='%.6f', header=False, index=False)
                        Predict_local_pd.to_excel(writer,sheet_name="predict",float_format='%.6f', header=False, index=False)
                print("   Local loss: Iterarion: {:03d} owner: {:03d} Test_Loss: {:.4f}".format(ec,i,loss_test_epoch_local))
            local_loss_quantile.append(loss_test_epoch_all)
            print("   Local all loss: Iterarion: {:03d} Test_Loss: {:.4f}".format(ec,loss_test_epoch_all))

        global_model.eval()
        with torch.no_grad():
            loss_test_global = 0
            for data in test_owner_all_loader:
                loss_function = QuantileLoss(quantile_list)
                prediction_all = global_model(tensor_adj_global,data)
                test_global_label = data["flow_y"].to(config.DEVICE)
                loss_global = loss_function(prediction_all,test_global_label)
                loss_test_global += loss_global.item()
            print("       Global loss: Iterarion: {:03d} Test_Loss: {:.4f}".format(ec,loss_test_global))
            global_loss_quantile.append(loss_test_global)
        tags = ["Local_loss", "Global_loss"]
        logger.add_scalar(tags[0], loss_test_epoch_all, ec)
        logger.add_scalar(tags[1], loss_test_global, ec)

    local_loss_quantile_array = np.array(local_loss_quantile)
    global_loss_quantile_array = np.array(global_loss_quantile)

    local_loss_quantile_pd = pd.DataFrame(local_loss_quantile_array)
    global_loss_quantile_pd = pd.DataFrame(global_loss_quantile_array)

    file_name_local_loss = r'F:\local_loss.xlsx'
    with pd.ExcelWriter(file_name_local_loss) as writer:
        local_loss_quantile_pd.to_excel(writer,sheet_name="local_all_loss",float_format='%.6f', header=False, index=False)

    file_name_global_loss = r'F:\global_loss.xlsx'
    with pd.ExcelWriter(file_name_global_loss) as writer:
        global_loss_quantile_pd.to_excel(writer,sheet_name="global_loss",float_format='%.6f', header=False, index=False)

