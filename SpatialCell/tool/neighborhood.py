# from Allen et al., 2023

# for each cell compute statistics of neighbors within radius
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from tqdm import tqdm
import pandas as pd

def compute_neighborhood_stats(pos, labels, radius=None,encoder = None):
    if radius is None:
        radius = np.mean(np.linalg.norm(pos[1:] - pos[:-1], axis=1))/100
        print(f'Using radius of {radius}')
    # record labels as numbers
    if encoder is None:
        encoder = LabelEncoder()
    labels_quant = encoder.fit_transform(labels)
    # for each cell, look up neighbors
    kdtree = KDTree(pos)
    nbors_idx, nbors_dist = kdtree.query_radius(pos, r=radius, return_distance=True)
    nbor_stats = np.zeros((pos.shape[0], len(np.unique(labels_quant))))

    for i in range(pos.shape[0]):
        curr_nbors_idx = np.sort(nbors_idx[i][nbors_dist[i]>0])#[1:]
        curr_nbors_labels = labels_quant[curr_nbors_idx]
        for j in curr_nbors_labels:
            nbor_stats[i,j] += 1
    nbor_stats = pd.DataFrame(nbor_stats, columns=np.unique(labels), index=range(pos.shape[0]))
    return nbor_stats,encoder

def neighborhood(adata,batch = None,groupby = 'cell_type',radius = None):
    if batch is None:
        batch = 'batch'
        adata.obs[batch] = '_virtual'
    for key in tqdm(adata.obs[batch].unique()):
        sub = adata[adata.obs[batch] == key]
        pos = sub.obsm['spatial']
        labels = sub.obs[groupby].values
        if 'nbor_label' in adata.uns.keys():
            nbor_stats,_ = compute_neighborhood_stats(pos, labels, radius,encoder)
        else:
            nbor_stats,encoder = compute_neighborhood_stats(pos, labels, radius)
            adata.uns['encorder'] = encoder
            adata.uns['nbor_label'] = nbor_stats.columns
            adata.obsm['nbor_counts'] = np.zeros((adata.shape[0],
                                                  len(adata.obs['cell_type'].cat.categories)))
        indices = adata.obs_names.get_indexer(sub.obs_names)
        
        adata.obsm['nbor_counts'][indices,:] = np.array(nbor_stats)
    if key == '_virtual':
        adata.obs.pop('batch', None)
