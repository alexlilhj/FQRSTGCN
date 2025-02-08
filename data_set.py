import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

def get_flow_data_owner_all(feature_file,label_file):
    data = pd.read_excel(feature_file)
    label = pd.read_excel(label_file)
    sample_data = np.array(data)
    label_data = np.array(label)
    sample_data= sample_data.reshape(100000,32,2)
    sample_data = sample_data.transpose(1,0,2)
    label_data=label_data
    return sample_data, label_data

def get_flow_data_owner_1(feature_file,label_file):
    data = pd.read_excel(feature_file)
    label = pd.read_excel(label_file)
    sample_data = np.array(data)
    label_data = np.array(label)
    sample_data= sample_data.reshape(100000,16,2)
    sample_data = sample_data.transpose(1,0,2)
    label_data=label_data
    return sample_data, label_data


def get_flow_data_owner_2(feature_file,label_file):
    data = pd.read_excel(feature_file)
    label = pd.read_excel(label_file)
    sample_data = np.array(data)
    label_data = np.array(label)
    sample_data= sample_data.reshape(100000,4,2)
    sample_data = sample_data.transpose(1,0,2)
    label_data=label_data
    return sample_data, label_data


def get_flow_data_owner_3(feature_file,label_file):
    data = pd.read_excel(feature_file)
    label = pd.read_excel(label_file)
    sample_data = np.array(data)
    label_data = np.array(label)
    sample_data= sample_data.reshape(100000,11,2)
    sample_data = sample_data.transpose(1,0,2)
    label_data=label_data
    return sample_data, label_data

class LoadData_all(Dataset):

    def __init__(self, x_path_all, y_path_all, num_nodes, divide_data, time_interval, history_length, train_mode):


        self.x_path_all = x_path_all
        self.y_path_all = y_path_all
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_data[0]
        self.test_days = divide_data[1]
        self.history_length = history_length
        self.time_interval = time_interval
        self.data = get_flow_data_owner_all(x_path_all,y_path_all)
        self.one_day_length = int(24*60/self.time_interval)


    def __len__(self):

        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days*self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData_owner_1.slice_data(self.data[0], self.data[1], self.history_length, index, self.train_mode)

        data_x = LoadData_owner_1.to_tensor(data_x)
        data_y = LoadData_owner_1.to_tensor(data_y)

        return{"flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data_i, data_o, history_length, index,train_mode):

        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train mode: [{}] is not defined".format(train_mode))

        data_x = data_i[:, start_index:end_index]
        data_y = data_o[end_index-1]

        return data_x, data_y

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)




class LoadData_owner_1(Dataset):

    def __init__(self, x_path_1, y_path_1, num_nodes, divide_data, time_interval, history_length, train_mode):


        self.x_path_1 = x_path_1
        self.y_path_1 = y_path_1
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_data[0]
        self.test_days = divide_data[1]
        self.history_length = history_length
        self.time_interval = time_interval
        self.data = get_flow_data_owner_1(x_path_1,y_path_1)
        self.one_day_length = int(24*60/self.time_interval)


    def __len__(self):

        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days*self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData_owner_1.slice_data(self.data[0], self.data[1], self.history_length, index, self.train_mode)

        data_x = LoadData_owner_1.to_tensor(data_x)
        data_y = LoadData_owner_1.to_tensor(data_y)

        return{"flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data_i, data_o, history_length, index,train_mode):

        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train mode: [{}] is not defined".format(train_mode))

        data_x = data_i[:, start_index:end_index]
        data_y = data_o[end_index-1]

        return data_x, data_y

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)





class LoadData_owner_2(Dataset):

    def __init__(self, x_path_2, y_path_2, num_nodes, divide_data, time_interval, history_length, train_mode):


        self.x_path_2 = x_path_2
        self.y_path_2 = y_path_2
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_data[0]
        self.test_days = divide_data[1]
        self.history_length = history_length
        self.time_interval = time_interval
        self.data = get_flow_data_owner_2(x_path_2,y_path_2)
        self.one_day_length = int(24*60/self.time_interval)


    def __len__(self):

        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days*self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData_owner_1.slice_data(self.data[0], self.data[1], self.history_length, index, self.train_mode)

        data_x = LoadData_owner_1.to_tensor(data_x)
        data_y = LoadData_owner_1.to_tensor(data_y)

        return{"flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data_i, data_o, history_length, index,train_mode):

        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train mode: [{}] is not defined".format(train_mode))

        data_x = data_i[:, start_index:end_index]
        data_y = data_o[end_index-1]

        return data_x, data_y

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)






class LoadData_owner_3(Dataset):

    def __init__(self, x_path_3, y_path_3, num_nodes, divide_data, time_interval, history_length, train_mode):


        self.x_path_3 = x_path_3
        self.y_path_3 = y_path_3
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_data[0]
        self.test_days = divide_data[1]
        self.history_length = history_length
        self.time_interval = time_interval
        self.data = get_flow_data_owner_3(x_path_3,y_path_3)
        self.one_day_length = int(24*60/self.time_interval)


    def __len__(self):

        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):
        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days*self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData_owner_1.slice_data(self.data[0], self.data[1], self.history_length, index, self.train_mode)

        data_x = LoadData_owner_1.to_tensor(data_x)
        data_y = LoadData_owner_1.to_tensor(data_y)

        return{"flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data_i, data_o, history_length, index,train_mode):

        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train mode: [{}] is not defined".format(train_mode))

        data_x = data_i[:, start_index:end_index]
        data_y = data_o[end_index-1]

        return data_x, data_y

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)



if __name__ == '__main__':
    train_data = LoadData_owner_1(x_path_1="input_1.xlsx", y_path_1="final_data_1_output.xlsx", num_nodes=16, divide_data=[170, 48],
                                  time_interval=15, history_length=21, train_mode="train")

    train_data[0]["flow_x"]
    print(len(train_data))
    print(train_data[0]["flow_x"].size())
    print(train_data[0]["flow_y"].size())
