
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

# from Allen et al., 2023

# for each cell compute statistics of neighbors within radius
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
            encoder = LabelEncoder()
            encoder.classes_ = adata.uns['encorder']["classes_"]
            nbor_stats,_ = compute_neighborhood_stats(pos, labels, radius,encoder)
        else:
            nbor_stats,encoder = compute_neighborhood_stats(pos, labels, radius)
            encoder_dict = {"classes_": encoder.classes_.tolist()}
            adata.uns['encorder'] = encoder_dict
            adata.uns['nbor_label'] = list(nbor_stats.columns)
            adata.obsm['nbor_counts'] = np.zeros((adata.shape[0],
                                                  len(adata.obs['cell_type'].cat.categories)))
        indices = adata.obs_names.get_indexer(sub.obs_names)
        nbor_stats = nbor_stats.div(nbor_stats.sum(axis=1), axis=0)
        nbor_stats = nbor_stats.fillna(0)

        adata.obsm['nbor_counts'][indices,:] = np.array(nbor_stats)
    if key == '_virtual':
        adata.obs.pop('batch', None)


def get_corrdf(adata,satge_order,batch ='sample',groupby = 'cell_type',stage = 'stage'):
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
    return heatmap_data,p_value_data


def corr_pl(adata,satge_order,batch ='sample',groupby = 'cell_type',stage = 'stage' , ax = None,figsize=(10, 8),cmap = 'coolwarm'):

    heatmap_data,p_value_data = get_corrdf(adata,satge_order,batch,groupby,stage)
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
    

def aic_bic(adata , extent = (1,10)):
    random_state = 42
    data = adata.obsm['nbor_counts']
    n_samples = int(data.shape[0] / 10)
    np.random.seed(random_state)
    indices = np.random.choice(data.shape[0], size=n_samples, replace=False)
    X = data[indices, :]
        # 设置最大簇数

    # 初始化 AIC 和 BIC 的存储列表
    aic_values = []
    bic_values = []
    components = np.arange(extent[0], extent[1], 1)
    # 计算每个簇数对应的 AIC 和 BIC
    for n in components:
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(X)
        aic_values.append(gmm.aic(X))
        bic_values.append(gmm.bic(X))

    # 可视化 AIC 和 BIC
    plt.figure(figsize=(8, 5))
    plt.plot(components, aic_values, label='AIC', marker='o')
    plt.plot(components, bic_values, label='BIC', marker='s')
    plt.xticks(components)
    plt.title("AIC and BIC for Gaussian Mixture Models")
    plt.xlabel("Number of Components")
    plt.ylabel("Information Criterion Value")
    plt.legend()
    plt.grid()
    plt.show()
    return aic_values,bic_values