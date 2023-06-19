import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from options import arg_parameter
from load_data import *
from model import *
from utils import *
from federated import MetrFedEngine
from aggregator import parameter_aggregate, read_out


def main(args):
    args.device = torch.device(args.device)

    '''
    读取传感器ID文件内容，将各个独立的传感器ID输出为一个字符串列表。strip()清楚字符串前后的空白字符、split(",")以逗号为分隔符将整个字符串拆分成一个个独立的传感器ID字符串
    '''
    print("Prepare data and model")
    train_iter, test_iter, val_iter, G, n_route, adj_mx = load_data(args)
    train, test, val = list(train_iter), list(test_iter), list(val_iter)
    print(len(train), len(test), len(val))

    G = G.to(args.device)
    model = STGCN_WAVE(
        args.channels, args.window, n_route, G, 0, args.num_layers, args.device, args.control_str
    ).to(args.device)    # 使用时空图卷积网络和波动预测模型

    print("Parameter holders")
    w_server, w_local = model.get_state()
    w_server = [w_server] * args.clients
    w_local = [w_local] * args.clients
    global_model = copy.deepcopy(w_server)
    personalized_model = copy.deepcopy(w_local)
    # print(len(global_model))
    # print(len(personalized_model))

    server_state = None
    clients_states = [None] * args.clients

    print2file(str(args), args.logDir, True)
    nParams = sum([p.nelement() for p in model.parameters()])
    print2file('Number of model parameters is ' + str(nParams), args.logDir, True)

    print("Start Training...")
    num_collaborator = max(int(args.client_frac * args.clients), 1)
    for com in range(1, args.com_round + 1):
        selected_user = np.random.choice(range(args.clients), num_collaborator, replace=False)
        train_time = []
        train_loss = []
        MAE, MAPE, RMSE = 0, 0, 0

        for c in selected_user:
            engine = MetrFedEngine(args, train_iter, global_model[c], personalized_model[c],
                                   w_local[c], {}, c, 0, "Train", server_state, clients_states[c])
            outputs = engine.run()

            w_server[c] = copy.deepcopy(outputs['params'][0])
            w_local[c] = copy.deepcopy(outputs['params'][1])
            train_time.append(outputs["time"])
            train_loss.append(outputs["loss"])
            MAE = outputs["MAE"]
            MAPE = outputs["MAPE"]
            RMSE = outputs["RMSE"]
            clients_states[c] = outputs["c_state"]

        mtrain_time = np.mean(train_time)
        mtrain_loss = np.mean(train_loss)

        log = 'Communication Round: {:03d}, Train Loss: {:.4f},' \
              ' MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}, Training Time: {:.4f}/com_round'
        print2file(log.format(com, mtrain_loss, MAE, MAPE, RMSE, mtrain_time),
                   args.logDir, True)

        t1 = time.time()
        personalized_model, clients_states, server_state = \
            parameter_aggregate(args, adj_mx, w_server, global_model, server_state, clients_states, selected_user)
        t2 = time.time()
        log = 'Communication Round: {:03d}, Aggregation Time: {:.4f} secs'
        print2file(log.format(com, (t2 - t1)), args.logDir, True)

        global_model = read_out(personalized_model, args.device)

        if com % args.valid_freq == 0:
            single_vtime = []
            single_vloss = []

            all_vtime = []
            all_vloss = []
            all_vMAE, all_vMAPE, all_vRMSE = 0, 0, 0

            for c in range(args.clients):
                batch_time = []
                batch_loss = []
                batch_MAE, batch_MAPE, batch_RMSE = 0, 0, 0

                for iter in test_iter:
                    tengine = MetrFedEngine(args, copy.deepcopy(iter), personalized_model[c], personalized_model[c],
                                            w_local[c], {}, c, 0, "Test", server_state, clients_states[c])
                    outputs = tengine.run()

                    batch_time.append(outputs["time"])
                    batch_loss.append(outputs["loss"])
                    batch_MAE = outputs["MAE"]
                    batch_MAPE = outputs["MAPE"]
                    batch_RMSE = outputs["RMSE"]

                single_vtime.append(batch_time[c])
                single_vloss.append(batch_loss[c])

                all_vtime.append(np.mean(batch_time))
                all_vloss.append(np.mean(batch_loss))
                all_vMAE = batch_MAE
                all_vMAPE = batch_MAPE
                all_vRMSE = batch_RMSE

            single_log = 'SingleValidation Round: {:03d}, Valid Loss: {:.4f}, Test Time: {:.4f}/epoch'
            print2file(single_log.format(com, np.mean(single_vloss), np.mean(single_vtime)), args.logDir, True)

            all_log = 'AllValidation Round: {:03d}, Valid Loss: {:.4f}, ' \
                      'Valid MAE: {:.6f}, Valid MAPE: {:.6f}, Valid RMSE: {:.6f}, Test Time: {:.4f}/epoch'
            print2file(all_log.format(com, np.mean(all_vloss), all_vMAE, all_vMAPE, all_vRMSE,
                                      np.mean(all_vtime)), args.logDir, True)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    option = arg_parameter()
    initial_environment(option.seed)
    main(option)

    print("Everything so far so good....")
