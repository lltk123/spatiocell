from sklearn.neighbors import NearestNeighbors
import numpy as np

distance_threshold = 8

def calculate_colocation_frequency(data,target1,target2):
    # 筛选NK和T细胞
    nk_cells = data[data['label']== target1][['x', 'y']]
    t_cells = data[data['label'] == target2][['x', 'y']]
    if nk_cells.shape[0]==0 :
        return -1
    if  t_cells.shape[0]==0 :
        return -2
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(t_cells)
    distances, _ = nbrs.kneighbors(nk_cells)

    colocalized_pairs = np.sum(distances < distance_threshold)
    
    # 计算共定位频率
    colocation_frequency = colocalized_pairs / len(nk_cells)
    return colocation_frequency
