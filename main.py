from data_set import LoadData_owner_1, LoadData_owner_2, LoadData_owner_3, LoadData_all
from torch.utils.data import DataLoader
from adj_create import create_adj_global,create_adj_owneri,normalize_data,create_adj_all
import config
from fedsage import train_fedSage
from aggregates import Aggregate
from phe import paillier


global_pub_key, global_priv_key = paillier.generate_paillier_keypair()

train_data_owner_1 = LoadData_owner_1(x_path_1="input_1.xlsx", y_path_1="final_data_1_output.xlsx", num_nodes=16, divide_data=[800, 241],
                                        time_interval=15, history_length=config.input_length, train_mode="train")
train_owner_1_loader = DataLoader(train_data_owner_1,batch_size=config.batch_size, shuffle=True, num_workers=0)


test_data_owner_1 = LoadData_owner_1(x_path_1="input_1.xlsx", y_path_1="final_data_1_output.xlsx", num_nodes=16, divide_data=[800, 241],
                                     time_interval=15, history_length=config.input_length, train_mode="test")
test_owner_1_loader = DataLoader(test_data_owner_1,batch_size=config.batch_size, shuffle=False, num_workers=0)


train_data_owner_2 = LoadData_owner_2(x_path_2="input_2.xlsx", y_path_2="final_data_2_output.xlsx", num_nodes=4, divide_data=[800, 241],
                                      time_interval=15, history_length=config.input_length, train_mode="train")
train_owner_2_loader = DataLoader(train_data_owner_2,batch_size=config.batch_size, shuffle=True, num_workers=0)

test_data_owner_2 = LoadData_owner_2(x_path_2="input_2.xlsx", y_path_2="final_data_2_output.xlsx", num_nodes=4, divide_data=[800, 241],
                                     time_interval=15, history_length=config.input_length, train_mode="test")
test_owner_2_loader = DataLoader(test_data_owner_2,batch_size=config.batch_size, shuffle=False, num_workers=0)


train_data_owner_3 = LoadData_owner_3(x_path_3="input_3.xlsx", y_path_3="final_data_3_output.xlsx", num_nodes=11, divide_data=[800, 241],
                                      time_interval=15, history_length=config.input_length, train_mode="train")
train_owner_3_loader = DataLoader(train_data_owner_3,batch_size=config.batch_size, shuffle=True, num_workers=0)

test_data_owner_3 = LoadData_owner_3(x_path_3="input_3.xlsx", y_path_3="final_data_3_output.xlsx", num_nodes=11, divide_data=[800, 241],
                                     time_interval=15, history_length=config.input_length, train_mode="test")
test_owner_3_loader = DataLoader(test_data_owner_3,batch_size=config.batch_size, shuffle=False, num_workers=0)


train_data_owner_all = LoadData_all(x_path_all="input_all.xlsx", y_path_all="final_data_all_output.xlsx", num_nodes=32, divide_data=[800, 241],
                                      time_interval=15, history_length=config.input_length, train_mode="train")
train_owner_all_loader = DataLoader(train_data_owner_all,batch_size=config.batch_size, shuffle=True, num_workers=0)

test_data_owner_all = LoadData_all(x_path_all="input_all.xlsx", y_path_all="final_data_all_output.xlsx", num_nodes=32, divide_data=[800, 241],
                                     time_interval=15, history_length=config.input_length, train_mode="test")
test_owner_all_loader = DataLoader(test_data_owner_all,batch_size=config.batch_size, shuffle=False, num_workers=0)


traindata_loader = []
traindata_loader.append(train_owner_1_loader)
traindata_loader.append(train_owner_2_loader)
traindata_loader.append(train_owner_3_loader)


testdata_loader = []
testdata_loader.append(test_owner_1_loader)
testdata_loader.append(test_owner_2_loader)
testdata_loader.append(test_owner_3_loader)


local_models = []
tensor_adj_all = create_adj_all()
tensor_adj_global = create_adj_global()

tensor_adj = []
for owner_i in range(config.num_owners):
    tensor_adj.append(create_adj_owneri(owner_i))

for owner_i in range(config.num_owners):
    train_data_loader_owneri = traindata_loader[owner_i]
    adj_owneri = tensor_adj[owner_i]
    local_model = Aggregate(owner_i,train_data_loader_owneri,adj_owneri,config.input_length,config.quantile_list_num)
    local_models.append(local_model)

quantile_list = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
train_fedSage(local_models,config.num_owners,testdata_loader,config.communication_num,tensor_adj_global,test_owner_all_loader,quantile_list,global_pub_key, global_priv_key)


