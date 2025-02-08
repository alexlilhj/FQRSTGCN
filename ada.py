
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

class ada:
    def __init__(self,
                loss: nn.Module,
                train_data: List[Tuple],
                adj_owneri,
                batch_size: int,
                eta: float = 1.0,
                device: str = "cuda",
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:

        self.loss = loss
        self.train_data = train_data
        self.adj = adj_owneri
        self.batch_size = batch_size
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:

        params_g = list(global_model.parameters())
        params = list(local_model.parameters())


        if torch.sum(params_g[0] - params[0]) == 0:
            return

        for param, param_g in zip(params, params_g):
            param.data = param_g.data.clone()

        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())


        params_p = params
        params_gp = params_g
        params_tp = params_t

        optimizer = torch.optim.SGD(params_tp, lr=0)


        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]


        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight


        losses = []
        cnt = 0
        while True:
            for data in self.train_data:

                y = data['flow_y'].to(self.device)
                optimizer.zero_grad()
                output = model_t(self.adj,data)
                loss_value = self.loss(output, y)
                loss_value.backward()

                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1


            if not self.start_phase:
                break


            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break

        self.start_phase = False


        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()


