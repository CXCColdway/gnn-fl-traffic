import argparse


def arg_parameter():
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--dataset', type=str, default='metr-la', help="name of dataset")
    parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    parser.add_argument('--reg', type=int, default=1, help='enable regulizer or not for local train')
    parser.add_argument('--com_round', type=int, default=3, help='Number of communication round to train.')
    parser.add_argument('--epoch', type=int, default=1, help='epoch for each communication round.')
    parser.add_argument('--client_epochs', type=int, default=1, help='epoch for each communication round.')
    parser.add_argument('--logDir', default='./log/,default.txt', help='Path for log info')
    parser.add_argument('--num_thread', type=int, default=5, help='number of threading to use for client training.')
    parser.add_argument('--dataaug', type=int, default=0, help='data augmentation')
    parser.add_argument('--evalall', type=int, default=0, help='use all or partial validation dataset for test')

    # Federated arguments
    parser.add_argument('--clients', type=int, default=2, help="number of users: K")
    parser.add_argument('--shards', type=int, default=2, help="each client roughly have 2 data classes")
    parser.add_argument('--serveralpha', type=float, default=1, help='server prop alpha')
    parser.add_argument('--serverbeta', type=float, default=0.3, help='personalized agg rate alpha')
    parser.add_argument('--deep', type=int, default=0, help='0: 1 layer only, 1: 2 layers, 3:full-layers')
    parser.add_argument('--agg', type=str, default='avg', help='averaging strategy')
    parser.add_argument('--dp', type=float, default=0.005, help='differential privacy')
    parser.add_argument('--epsilon', type=float, default=1, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    parser.add_argument('--sm', type=str, default='full', help='state mode, for baselines running')
    parser.add_argument('--layers', type=int, default=3, help='number of layers')
    parser.add_argument('--client_frac', type=float, default=0.5, help='the fraction of clients')

    # Graph Learning
    parser.add_argument('--subgraph_size', type=int, default=30, help='k')
    parser.add_argument('--adjalpha', type=float, default=3, help='adj alpha')
    parser.add_argument('--gc_epoch', type=int, default=10, help='')
    parser.add_argument('--adjbeta', type=float, default=0.05, help='update ratio')
    parser.add_argument('--edge_frac', type=float, default=1, help='the fraction of clients')

    # Others
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--print_every', type=int, default=100, help='')
    parser.add_argument('--save', type=str, default='./save/', help='save path')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--valid_freq', type=int, default=1, help='validation at every n communication round')
    parser.add_argument('--cl', type=int, default=1, help='whether to do curriculum learning')
    parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=128, help='end channels')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')
    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
    parser.add_argument('--runs', type=int, default=10, help='number of runs')

    parser.add_argument("--gnn_lr", default=0.001, type=float, help="learning rate")
    # parser.add_argument("--disablecuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--gnn_batch_size",
        type=int,
        default=10,
        help="batch size for training and validation (default: 50)",
    )
    parser.add_argument(
        "--gnn_epochs", type=int, default=1, help="epochs for training  (default: 50)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=9, help="number of layers"
    )
    parser.add_argument("--window", type=int, default=144, help="window length")
    parser.add_argument(
        "--sensorsfilepath",
        type=str,
        default="./data/sensor_graph/graph_sensor_ids.txt",
        help="sensors file path",
    )
    parser.add_argument(
        "--disfilepath",
        type=str,
        default="./data/sensor_graph/distances_la_2012.csv",
        help="distance file path",
    )
    parser.add_argument(
        "--tsfilepath", type=str, default="./data/metr-la.h5", help="ts file path"
    )
    parser.add_argument(
        "--savemodelpath",
        type=str,
        default="stgcnwavemodel.pt",
        help="save model path",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=5,
        help="how many steps away we want to predict",
    )  # 预学习步数
    parser.add_argument(
        "--control_str",
        type=str,
        default="TNTSTNTST",
        help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
    )  # 图神经网络结构模型控制器
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[1, 16, 32, 64, 32, 128],
        help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
    )  # 图神经网络各层节点数

    args = parser.parse_args()

    return args
