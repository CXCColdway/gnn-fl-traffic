import datetime
import time
import dgl
import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sensors2graph import *
from model import *
from utils import *
from load_data import *


class TorchOptimiser:
    def __init__(self, weights, optimizer, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())

    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k, v in self.opt_params.items()}

    def step(self):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def __repr__(self):
        return repr(self._opt)


def SGD(weights, lr=0, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    return TorchOptimiser(weights, torch.optim.SGD, lr=lr, momentum=momentum,
                          weight_decay=weight_decay, dampening=dampening,
                          nesterov=nesterov)


class MetrFedEngine:
    def __init__(self, args, dataiter, global_param, server_param, local_param,
                 outputs, cid, tid, mode, server_state, client_states):
        self.args = args
        self.dataiter = dataiter
        self.blocks = args.channels
        self.n_his = args.window
        self.n_route = 207
        self.num_layers = args.num_layers
        self.device = args.device
        self.control_str = args.control_str
        self.drop_prob = 0

        self.global_param = global_param
        self.server_param = server_param
        self.local_param = local_param
        self.server_state = server_state
        self.client_state = client_states

        self.client_id = cid
        self.outputs = outputs
        self.thread = tid

        self.mode = mode

        self.model = self.prepare_model()

        self.m1, self.m2, self.m3, self.reg1, self.reg2 = None, None, None, None, None

    def prepare_model(self):
        if self.args.dataset == "metr-la":
            with open(self.args.sensorsfilepath) as f:
                sensor_ids = f.read().strip().split(",")
            distance_df = pd.read_csv(self.args.disfilepath, dtype={"from": "str", "to": "str"})
            adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
            sp_mx = sp.coo_matrix(adj_mx)
            G = dgl.from_scipy(sp_mx)

            model = STGCN_WAVE(
                self.blocks, self.n_his, self.n_route, G, self.drop_prob, self.num_layers, self.device, self.control_str
            )
        else:
            print("Unknown model type ... ")
            model = None

        model.set_state(self.global_param, self.local_param)
        return model

    def run(self):
        self.model.to(self.args.device)
        output = self.client_run()
        self.free_memory()

        return output

    def client_run(self):
        '''
        lr_schedule = PiecewiseLinear([0, 5, self.args.client_epochs], [0, 0.4, 0.001])
        lr = lambda step: lr_schedule(step / len(self.dataiter)) / self.args.batch_size
        opt = SGD(trainable_params(self.model), lr=lr, momentum=0.9, weight_decay=5e-4 * self.args.batch_size, nesterov=True)
        '''

        t1 = time.time()
        mean_loss = []
        c_state = None
        MAE, MAPE, RMSE = 0, 0, 0

        if self.mode == "Train":
            for epoh in range(self.args.client_epochs):
                stats = self.batch_run_train(True)
                loss, MAE, MAPE, RMSE = stats
                mean_loss.append(loss)

        elif self.mode == "Test":
            stats = self.batch_run_test(False)
            loss, MAE, MAPE, RMSE = stats
            mean_loss.append(loss)
            # print('test loss:', mean_loss)

        time_cost = time.time() - t1
        # print('test time:', time_cost)
        log = self.mode + ' - Thread: {:03d}, Client: {:03d}. Average Loss: {:.4f},' \
                          ' MAE: {:.6f}, MAPE :{:.6f}, RMSE: {:.6f}, Total Time Cost: {:.4f}'
        self.logger(log.format(self.thread, self.client_id, np.mean(mean_loss), MAE, MAPE, RMSE,
                               time_cost), True)

        self.model.to("cuda")
        output = {"params": self.model.get_state(),
                  "time": time_cost,
                  "loss": mean_loss,
                  "MAE": MAE,
                  "MAPE": MAPE,
                  "RMSE": RMSE,
                  "client_state": self.client_state,
                  "c_state": c_state}

        return output

    def batch_run_train(self, training):  # 测试训练函数分离
        loss_list = []
        scaler = StandardScaler()
        self.model.train(training)
        loss = nn.MSELoss()
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        train_iter, test_iter, val_iter, G, n_route, adj_mx = load_data(args=self.args)
        for epoch in range(1, self.args.gnn_epochs + 1):
            l_sum, n = 0.0, 0
            self.model.train()
            for x, y in train_iter: # 测试数据集
                y_pred = self.model(x, self.args).view(len(x), -1)
                l = loss(y_pred, y)
                optimizer.zero_grad()  # 将此前的所有梯度清零，避免对后续训练产生影响
                l.backward()
                optimizer.step()  # 更新梯度参数,test不用做
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            scheduler.step()  # 更新学习率
            train_loss = l_sum / n
            loss_list.append(train_loss)
        MAE, MAPE, RMSE = evaluate_metric(self.model, train_iter, scaler, self.args)
        return loss_list, MAE, MAPE, RMSE

    def batch_run_test(self, training):  # 测试训练函数分离
        loss_list = []
        scaler = StandardScaler()
        self.model.train(training)
        loss = nn.MSELoss()
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        train_iter, test_iter, val_iter, G, n_route, adj_mx = load_data(args=self.args)
        for epoch in range(1, self.args.clients + 1):
            l_sum, n = 0.0, 0
            self.model.train()
            for x, y in test_iter: # 测试数据集
                y_pred = self.model(x, self.args).view(len(x), -1)
                l = loss(y_pred, y)
                optimizer.zero_grad()  # 将此前的所有梯度清零，避免对后续训练产生影响
                l.backward()
                optimizer.step()  # 更新梯度参数,test不用做
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
            scheduler.step()  # 更新学习率
            test_loss = l_sum / n
            loss_list.append(test_loss)
        MAE, MAPE, RMSE = evaluate_metric(self.model, test_iter, scaler, self.args)
        return loss_list, MAE, MAPE, RMSE

    def criterion(self, loss, mode):
        if self.args.agg == 'avg':
            pass
        elif self.args.reg > 0 and mode != "PerTrain" and self.args.clients != 1:
            self.m1 = sd_matrixing(self.model.get_state()[0]).reshape(1, -1).to(self.args.device)
            self.m2 = sd_matrixing(self.server_param).reshape(1, -1).to(self.args.device)
            self.m3 = sd_matrixing(self.global_param).reshape(1, -1).to(self.args.device)
            self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
            self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
            loss = loss + 0.3 * self.reg1 + 0.3 * self.reg2
        return loss

    def free_memory(self):
        if self.m1 is not None:
            self.m1.to("cuda")
        if self.m2 is not None:
            self.m2.to("cuda")
        if self.m3 is not None:
            self.m3.to("cuda")
        if self.reg1 is not None:
            self.reg1.to("cuda")
        if self.reg2 is not None:
            self.reg2.to("cuda")

        torch.cuda.empty_cache()

    def logger(self, buf, p=False):
        if p:
            print(buf)
        with open(self.args.logDir, 'a+') as f:
            f.write(str(datetime.datetime.now()) + '\t' + buf + '\n')
