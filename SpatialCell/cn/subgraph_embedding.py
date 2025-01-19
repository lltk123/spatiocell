import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

import numpy as np

def center_pooling(subg, k=1):
    if k > subg.k:
        raise ValueError('k should be smaller than the number of hops')

    # 存储所有 pooled features
    pooled_features = []

    # 避免多次调用 k_hop_subgraph，可以提前计算所有的子图节点
    subgraph_node_idxs = []
    cell_counts = {}
    for i in range(k):
        subgraph_node_idx, _, _, _ = k_hop_subgraph(subg.center, i + 1, subg.edge_index, num_nodes=subg.num_nodes)
        subgraph_node_idxs.append(subgraph_node_idx)
        cell_count = pd.Series(subg.label[subgraph_node_idx]).value_counts()
        for cell in np.unique(subg.label):
            if cell not in cell_count:
                cell_count[cell] = 0
        cell_counts[i] = cell_count
        if len(subgraph_node_idx) < 3:
            return None

    # 直接从节点索引中获取对应的 one-hot 编码特征
    for subgraph_node_idx in subgraph_node_idxs:
        subgraph_x = subg.onehot[subgraph_node_idx]
        pooled_features.append(subgraph_x.mean(axis=0))

    # 使用 np.concatenate 或 torch.cat 一次性合并所有 pooled_features
    pooled_features = np.concatenate(pooled_features, axis=0)  # 如果是使用 torch，可以用 torch.cat()

    return pooled_features,cell_counts


def embedding(mydata, k=3 , target = None):
    target_num = -1
    if target is not None:
        target = np.array([target]).reshape(-1,1)
        encoder = mydata.encorder
        target_num = encoder.transform(target).flatten()[0]
    sample_pool = []
    sample_cells = []
    for i in tqdm(mydata.adata.obs[mydata.batch].unique()):
        
        G = mydata[i]
        result = []
        cells = []
        if target_num == -1:
            subindex = range(G.num_nodes)
        else:
            encoder.inverse_transform(G.label)
            subindex = np.where(G.label == target_num)[0] 
        tqdm.write(f"Processing batch id = {i} , cell count = {len(subindex)}")
        for idx in subindex:
            subg = mydata.get_subgraph(i ,int(idx),k)
            pool,cell_counts = center_pooling(subg , k = k)
            if pool is not None:
                result.append(pool)
                cells.append(cell_counts)
        result = pd.DataFrame(result)
        result['index'] = subindex
        result['sample'] = i
        sample_pool += [result]
        sample_cells += [cells]
    return sample_pool,sample_cells