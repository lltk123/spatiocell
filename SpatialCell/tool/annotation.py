from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from .. import np, pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from skimage.filters import threshold_otsu



def get_index(data,conditions,method = 'otsu'):
    if type(data) == AnnData:
        data = pd.DataFrame(data.X.copy(),index = data.obs_names,columns = data.var_names)
    min_sample_size = 30000  # 最小样本阈值
    subset_condition = pd.Series(True, index=data.index)
    sub_df = data[subset_condition]
    thresholds = []
    for marker, sign in zip(conditions['marker'], conditions['sign']):
        if sub_df.shape[0]>min_sample_size:
            sub_df = sub_df.apply(
                lambda x: x if len(x) <= min_sample_size else x.sample(frac=0.3, random_state=0)
            )
        tmp_data = sub_df[marker].values.reshape(-1, 1)
        if method == 'GaussianMixture':
            gmm = GaussianMixture(n_components=2, random_state=0).fit(tmp_data)
            marker_means = gmm.means_.flatten()
            threshold = np.mean(marker_means[marker_means > np.median(marker_means)])

        elif method == 'otsu':
            tmp_data = sub_df[marker].values.flatten()
            threshold = threshold_otsu(tmp_data)

        thresholds.append(threshold)
        if sign == '+':
            subset_condition &= data[marker].values.flatten() > threshold  # 大于条件
        elif sign == '-':
            subset_condition &= data[marker].values.flatten()  <= threshold   # 小于条件

        sub_df = data[subset_condition]
    subset_indices = data[subset_condition].index
    return subset_indices,thresholds

def cellphenotype(adata,condition,obs_name,cell_type,plot = False,method = 'otsu'):
    """
    根据给定的条件对每个细胞进行标注。

    参数:
        adata (AnnData): 输入数据。
        condition (dict): 包含两个部分:
            - 'marker' (list): 要考虑的通道名，与result列名对应。
            - 'sign' (list): 对应每个marker的要求 ('+' 或 '-')。

    返回:
        pd.Series: 一个布尔值的Series，表示每个细胞是否符合条件。
    """
    result,thresholds = get_index(adata,condition,method = method)
    if obs_name not in adata.obs.keys():
        adata.obs[obs_name] = 'unknown'
    adata.obs[obs_name] = adata.obs[obs_name].astype('category')
    if cell_type not in adata.obs[obs_name].cat.categories:
        adata.obs[obs_name] = adata.obs[obs_name].cat.add_categories(cell_type)
    adata.obs.loc[result,obs_name] = cell_type
    
    if plot:
        data = pd.DataFrame(adata.X.copy(),index = adata.obs_names,columns = adata.var_names)
        _,ax = plt.subplots(1,len(condition['marker']),figsize=(5*len(condition['marker']),5))
        if len(condition['marker']) == 1:
            ax = [ax]
        for i, marker in enumerate(condition['marker']):
            sns.kdeplot(data[marker], ax=ax[i], bw_adjust=2)
            ax[i].axvline(thresholds[i], color='r')
            ax[i].set_title(marker)
        plt.show()

