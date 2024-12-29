from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from .. import np, pd

def get_index(adata):
    """
    对样本的每一列进行计算，标注每一行在每一列的数据中应当为'+'或是‘-’，其中平均值较高的用'+'表示。

    参数:
    adata (AnnData): 输入数据。
    obs_name (str): 结果列名前缀。

    返回:
    DataFrame: 标注后的数据。
    """
    
    data = pd.DataFrame(adata.X.copy(),columns=adata.var_names , index = adata.obs_names)
    result = pd.DataFrame(index=data.index)

    for column in data.columns:
        # 使用高斯混合模型拟合数据
        gmm = GaussianMixture(n_components=2, random_state=0).fit(data[[column]])
        marker_means = gmm.means_.flatten()
        # 计算阈值
        threshold = np.mean(marker_means[marker_means > np.median(marker_means)])
        
        # 标注每一行在每一列的数据中应当为'+'或是‘-’
        result[f"{column}"] = np.where(data[column] > threshold, '+', '-')
    
    return result

def label_cells_by_condition(result, condition):
    """
    根据给定的条件对每个细胞进行标注。

    参数:
        result (pd.DataFrame): 包含多个列，每列表示某个marker的+/−情况。
        condition (dict): 包含两个部分:
            - 'marker' (list): 要考虑的通道名，与result列名对应。
            - 'sign' (list): 对应每个marker的要求 ('+' 或 '-')。

    返回:
        pd.Series: 一个布尔值的Series，表示每个细胞是否符合条件。
    """
    # 确保条件的长度一致
    if len(condition['marker']) != len(condition['sign']):
        raise ValueError("The length of 'marker' and 'sign' in condition must be the same.")
    
    # 初始化符合条件的布尔值
    match = pd.Series([True] * len(result), index=result.index)
    
    # 按条件逐列判断
    for marker, sign in zip(condition['marker'], condition['sign']):
        if sign == '+':
            match &= result[marker] == '+'
        elif sign == '-':
            match &= result[marker] == '-'
        else:
            raise ValueError("Sign must be either '+' or '-'")
    match.index = result.index
    return match.astype(bool)

def cellphenotype(adata,condition,obs_name,cell_type,):
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
    result = get_index(adata)
    if obs_name not in adata.obs.keys():
        adata.obs[obs_name] = 'unknown'
    adata.obs[obs_name] = adata.obs[obs_name].astype('category')
    adata.uns['sign'] = result
    label = label_cells_by_condition(result, condition)
    adata.obs[obs_name] = adata.obs[obs_name].cat.add_categories(cell_type)
    adata.obs.loc[label,obs_name] = cell_type

