import numpy as np


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return: adjacency matrix
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf  # 将所有元素均赋值为正无穷进行占位
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
    '''
    Fills cells in the matrix with distances.
    如果其中任意一个传感器 ID 没有对应的节点索引，则跳过当前行的处理。
    如果两个传感器 ID 都有对应的节点索引，则将距离数据框中对应行的距离值 row[2] 赋值给距离矩阵 dist_mx 中对应的元素。
    '''
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    '''
    Calculates the standard deviation as theta.
    首先计算距离的标准差，然后依据公式w_{i,j} = \exp(-\frac{(d_{i,j}/\sigma)^2}{2})计算得到毗连矩阵
    '''
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx
