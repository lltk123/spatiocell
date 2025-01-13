# from Allen et al., 2023

# for each cell compute statistics of neighbors within radius
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from tqdm import tqdm
import pandas as pd

def compute_neighborhood_stats(pos, labels, radius=None):
    if radius is None:
        radius = np.mean(np.linalg.norm(pos[1:] - pos[:-1], axis=1))/100
        print(f'Using radius of {radius}')
    # record labels as numbers
    labels_quant = LabelEncoder().fit_transform(labels)
    # for each cell, look up neighbors
    kdtree = KDTree(pos)
    nbors_idx, nbors_dist = kdtree.query_radius(pos, r=radius, return_distance=True)
    nbor_stats = np.zeros((pos.shape[0], len(np.unique(labels_quant))))

    for i in tqdm(range(pos.shape[0])):
        curr_nbors_idx = np.sort(nbors_idx[i][nbors_dist[i]>0])#[1:]
        curr_nbors_labels = labels_quant[curr_nbors_idx]
        for j in curr_nbors_labels:
            nbor_stats[i,j] += 1
    nbor_stats = pd.DataFrame(nbor_stats, columns=np.unique(labels), index=range(pos.shape[0]))

    return nbor_stats
