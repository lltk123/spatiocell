# from Allen et al., 2023

# for each cell compute statistics of neighbors within radius
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

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
    for key in tqdm(adata.obs[batch].unique(), leave=True):
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

def corr_pl(adata,satge_order,batch ='sample',groupby = 'cell_type',stage = 'stage' , ax = None,figsize=(10, 8),cmap = 'coolwarm'):
    if 'nbor_counts' not in adata.obsm.keys():
         raise ValueError("Dont have neighborhood counts, please run neighborhood first")
    
    figdf = pd.DataFrame(adata.obsm['nbor_counts'], columns = adata.uns['nbor_label'] ,index = adata.obs_names)
    figdf['cell_type'] = adata.obs[groupby]
    figdf['stage'] = adata.obs[stage]
    figdf['sample'] = adata.obs[batch]
    figdf = figdf.groupby(by = ['cell_type','stage','sample']).mean()
    figdf = figdf.reset_index()
    figdf.dropna(inplace = True)
    figdf['stage'] = figdf['stage'].map(satge_order)
    corrdf = pd.DataFrame()

    for i in figdf['cell_type'].unique():
        subdf = figdf[figdf['cell_type'] == i]
        correlations = {}
        for column in subdf.columns:
            if column not in ['cell_type','stage','sample']:  # 排除 'col1' 自己与自己相关
                corr, p_value = pearsonr(subdf['stage'], subdf[column])
                correlations[column] = {'correlation': corr, 'p_value': p_value}
        correlations = pd.DataFrame(correlations).T
        correlations['center_cell'] = i
        corrdf = pd.concat([corrdf,correlations])
    corrdf['cell_type'] = corrdf.index
    corr_value = corrdf.pivot(columns='cell_type',index ='center_cell', values='correlation')
    corr_p = corrdf.pivot(columns='cell_type',index ='center_cell', values='p_value')
    adata.uns['stage_corr_nborh'] = corrdf
    heatmap_data = corr_value
    p_value_data = corr_p

    if ax == None:
        fig, ax = plt.subplots(figsize=figsize,)
    sns.heatmap(heatmap_data, annot=False, cmap=cmap, ax=ax , )

    # 添加显著性标记
    for i in range(len(p_value_data)):
        for j in range(len(p_value_data.columns)):
            p_value = p_value_data.iloc[i, j]
            if p_value <= 0.001:
                ax.text(j + 0.5, i + 0.5, '***', ha='center', va='center', color='white', fontsize=12)
            elif p_value <= 0.01:
                ax.text(j + 0.5, i + 0.5, '**', ha='center', va='center', color='white', fontsize=12)
            elif p_value <= 0.05:
                ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center', color='white', fontsize=12)
    return fig,ax
    

