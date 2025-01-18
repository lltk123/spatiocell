import torch
from torch_geometric.data import Data,Dataset
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import Delaunay
import numpy as np
from torch_geometric.utils import to_undirected, k_hop_subgraph
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

def calculate_edge_weights(pos, edge_index):
        # 使用广播来计算欧几里得距离
    node_i_pos = pos[edge_index[0]]  # 获取所有边的第一个节点的位置
    node_j_pos = pos[edge_index[1]]  # 获取所有边的第二个节点的位置

    # 计算节点间的差距并获取欧几里得距离
    distances = torch.sqrt(torch.sum((node_i_pos - node_j_pos) ** 2, dim=1))
    return torch.tensor(distances)

def get_Graph(points , labels):
    triangulation = Delaunay(points)
    edges = set()  
    for simplex in triangulation.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    edge_index = np.array(list(edges)).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weights = calculate_edge_weights(points, edge_index)

    percentile_99 = torch.quantile(edge_weights, 0.99)
    mask = edge_weights > percentile_99

    filtered_edge_index = edge_index[:, ~mask] 
    filtered_edge_attr = edge_weights[~mask]
    filtered_data = Data(edge_index=filtered_edge_index,
                         edge_attr=filtered_edge_attr,
                         pos=points,
                         label = labels)
    filtered_data.edge_index = to_undirected(filtered_data.edge_index)
    
    return filtered_data
def get_k_hop_subgraph(node_idx, k, data):
    """
    获取从指定节点开始的k层邻域子图
    Args:
        node_idx (int): 指定的节点索引
        k (int): 邻域的层数
        data (torch_geometric.data.Data): 输入的图数据
    Returns:
        subgraph_data (torch_geometric.data.Data): K层邻域的子图
    """
    # 获取 k 层邻域的子图信息
    result = k_hop_subgraph(node_idx, k, data.edge_index, num_nodes=data.num_nodes)
    subgraph_node_idx, subgraph_edge_index, _,_ = result
    original_node_idx = data.edge_index[0, subgraph_edge_index[0]]
    return subgraph_node_idx , subgraph_edge_index  , original_node_idx

class GraphDataset(Dataset):
    def __init__(self, adata, batch = 'sample',groupby = 'cell_type', spatial = 'spatial'):
        super(GraphDataset, self).__init__()
        self.adata = adata
        self.batch = batch
        self.groupby = groupby
        self.spatial = spatial
        self.G = {}
        self.onehot_coder = OneHotEncoder(sparse=False)
        _ = self.onehot_coder.fit_transform(adata.obs[groupby].values.reshape(-1,1))
        encoder = LabelEncoder()
        if 'encorder' in self.adata.uns.keys():
            classes = self.adata.uns['encorder']['classes_']
            encoder.fit(classes)
        else:
            encoder.fit(adata.obs[groupby])
        self.encorder = encoder

    def __getitem__(self, idx,reset = False):
        if reset == True and idx in self.G.keys():
            return self.G[idx]
        sub = self.adata[self.adata.obs[self.batch] == idx]
        pos = torch.tensor(sub.obsm[self.spatial])
        labels = sub.obs[self.groupby]
        
        encoder = self.encorder
        labels_quant = encoder.fit_transform(labels)
        data = get_Graph(pos, labels_quant)

        onehot_coder = self.onehot_coder
        data.onehot = onehot_coder.transform(labels.values.reshape(-1,1))
        self.G[idx] = data
        return data
    
    def __len__(self):
        return len(self.adata.obs[self.batch].unique())
    
    def get_subgraph(self,idx, node_idx, k = 1):
        if idx not in self.G.keys():
            self.__getitem__(idx)
        G = self.G[idx]
        subgraph_node_idx , subgraph_edge_index , _  = get_k_hop_subgraph(node_idx, k, G)
        if len(subgraph_node_idx) == 0:
            print(f'No subgraph for {node_idx}')
            return None
        map_dict = {subgraph_node_idx.tolist()[i]:i for i in range(len(subgraph_node_idx))}

        keys = torch.tensor(range(len(subgraph_node_idx)))
        values = torch.tensor(subgraph_node_idx)
        result_tensor = torch.stack((keys, values))

        edge = torch.tensor([pd.Series(i).map(map_dict) for i in subgraph_edge_index] , dtype = torch.int32)
        subG = Data(edge_index = edge,
                    center = int(map_dict[node_idx]),
                    pos = G.pos[subgraph_node_idx],
                    label = G.label[subgraph_node_idx] , 
                    onehot = G.onehot[subgraph_node_idx],
                    original_dir = result_tensor,
                    k = k)
        return subG

    def plG(self,G):
        fig, ax = plt.subplots()
        pos = G.pos
        encoder = self.encorder
        decoded_labels = pd.Series(encoder.inverse_transform(G.label))

        
        for i in G.edge_index.T:
            plt.plot(pos[i, 0], pos[i, 1], c='black', alpha=0.3)
        for i in decoded_labels.unique():
            plt.scatter(pos[decoded_labels == i, 0], pos[decoded_labels == i, 1],
                        label=i, s=20)
        plt.scatter(G.pos[G.center, 0], G.pos[G.center, 1],c = 'r' ,s = 30 , label = f'Center_{decoded_labels[G.center]}')

        plt.axis('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.grid(False)
        ax.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
        plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

        return fig,ax