import copy
import torch
import os
import pickle as pk
from utils import sd_matrixing
from data_util import normalize_adj
from GraphConstructor import GraphConstructor
from optimiser import FedProx
import numpy as np
from scipy import linalg


def parameter_aggregate(args, adj_mx, w_server, global_model, sever_state, client_states, active_idx):
    new_s_state = None
    new_c_state = [None] * args.clients
    if args.agg == 'avg' or args.agg == 'prox' or args.agg == "scaf":
        w_server = average_dic(w_server, args.device)
        w_sever = [w_server] * args.clients
        personalized_model = copy.deepcopy(w_sever)

    elif args.agg == 'att':
        w_sever = att_dic(w_server, global_model[0], args.device)
        w_sever = [w_sever] * args.clients
        personalized_model = copy.deepcopy(w_sever)

    elif args.agg == 'graph' or args.agg == 'graph_v2' or args.agg == 'graph_v3':
        personalized_model = graph_dic(w_server, adj_mx, args)

    elif args.agg == 'scaffold':
        new_s_state, new_c_state, = scaffold_update(sever_state, client_states, active_idx, args)
        w_sever = average_dic(w_server, args.device)
        w_sever = [w_sever] * args.clients
        personalized_model = copy.deepcopy(w_sever)

    else:
        personalized_model = None
        exit('Unrecognized aggregation.')

    return personalized_model, new_c_state, new_s_state


def average_dic(model_dic, device, dp=0.001):
    w_avg = copy.deepcopy(model_dic[0])
    for k in w_avg.keys():
        for i in range(1, len(model_dic)):
            w_avg[k] = w_avg[k].data.clone().cuda().detach() + model_dic[i][k].data.clone().cuda().detach()
        w_avg[k] = w_avg[k].data.clone().cuda().detach().div(len(model_dic)) + torch.mul(torch.randn(w_avg[k].shape), dp).to(device)
    return w_avg


def att_dic(w_clients, w_server, device, stepsize=1, metric=1, dp=0.001):
    w_next = copy.deepcopy(w_server)
    att, att_mat = {}, {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k]).cpu()
        att[k] = torch.zeros(len(w_clients)).cpu()
    for k in w_next.keys():
        for i in range(0, len(w_clients)):
            att[k][i] = torch.norm((w_server[k]-w_clients[i][k]).type(torch.float32), metric)
    for k in w_next.keys():
        att[k] = torch.nn.functional.softmax(att[k], dim=0)
    for k in w_next.keys():
        att_weight = torch.zeros_like(w_server[k])
        for i in range(0, len(w_clients)):
            datatype = w_server[k].dtype
            att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i].type(datatype))
        w_next[k] = w_server[k] - torch.mul(att_weight, stepsize) + torch.mul(torch.randn(w_server[k].shape), dp)
    return w_next


def graph_dic(models_dic, pre_adj, args):
    keys = []
    key_shapes = []
    param_metrix = []

    for model in models_dic:
        param_metrix.append(sd_matrixing(model).clone().detach())
    param_metrix = torch.stack(param_metrix)

    for key, param in models_dic[0].items():
        keys.append(key)
        key_shapes.append(list(param.data.shape))

    if args.agg == 'graph_v2' or args.agg == 'graph_v3':
        subgraph_size = min(args.subgraph_size, args.clients)
        adj_mx = generate_adj(param_metrix, args, subgraph_size).cpu().detach().numpy()
        adj_mx = normalize_adj(adj_mx)
        adj_mx = torch.tensor(adj_mx)
        if args.agg == 'graph_v3':
            adj_mx = (1 - args.adjbeta) * pre_adj + args.adjbeta * adj_mx
    else:
        adj_mx = torch.tensor(pre_adj)

    aggregated_param = torch.mm(adj_mx, param_metrix)   # 207*207与2*1974097无法相乘
    for i in range(args.layers - 1):
        aggregated_param = torch.mm(adj_mx, aggregated_param)
    new_param_matrix = (args.serveralpha * aggregated_param)

    for i in range(len(models_dic)):
        pointer = 0
        for k in range(len(keys)):
            num_p = 1
            for n in key_shapes[k]:
                num_p *= n
            models_dic[i][keys[k]] = new_param_matrix[i][pointer:pointer + num_p].reshape(key_shapes[k])
            pointer += num_p

    return models_dic


def scaffold_update(server_state, client_states, active_ids, args):
    active_clients = [client_states[i] for i in active_ids]
    c_delta = []
    cc = [client_state['c_i_delta'] for client_state in active_clients]
    for ind in range(len(server_state['c'])):
        c_delta.append(
            torch.mean(torch.stack([c_i_delta[ind].float() for c_i_delta in cc]), dim=0).to(server_state['c'][ind].dtype)
        )
    c_delta = tuple(c_delta)

    c = []
    for param_1, param_2 in zip(server_state['c'], c_delta):
        c.append(param_1 + param_2 * args.clients * args.client_frac / args.clients)

    c = tuple(c)

    new_server_state = {
        "global_round": server_state['global_round'] + 1,
        "c": c
    }

    new_client_state = [{
        "global_round": new_server_state["global_round"],
        "model_delta": None,
        "c_i": clinet["c_i"],
        "c_i_delta": None,
        "c": server_state["c"]
    } for clinet in client_states]

    return new_client_state, new_server_state


def generate_adj(param_metrix, args, subgraph_size):
    dist_metrix = torch.zeros((len(param_metrix), len(param_metrix)))
    for i in range(len(param_metrix)):
        for j in range(len(param_metrix)):
            dist_metrix[i][j] = torch.nn.functional.pairwise_distance(
                param_metrix[i][j].view(1, -1), param_metrix[j].view(1, -1), p=2).clone().detach()
    dist_metrix = torch.nn.functional.normalize(dist_metrix).to(args.device)

    gc = GraphConstructor(args.clients, subgraph_size, args.node_dim,
                          args.device, args.adjalpha).to(args.device)
    idx = torch.arange(args.clients).to(args.device)
    optimizer = torch.optim.SGD(gc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(args.gc_epoch):
        optimizer.zero_grad()
        adj = gc(idx)
        adj = torch.nn.functional.normalize(adj)

        loss = torch.nn.functional.mse_loss(adj, dist_metrix)
        loss.backward()
        optimizer.step()

    adj = gc.eval(idx).to("cuda")

    return adj


def read_out(personalized_models, device):
    global_model = average_dic(personalized_models, device, 0)
    return [global_model] * len(personalized_models)
