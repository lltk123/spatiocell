import pandas as pd
import numpy as np
import os
from anndata import AnnData

def normalize_value(x, q):
    return np.arcsinh(x / (5 * q))
def zscore(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data

def normal(adata,chanel):
    df = pd.DataFrame(adata.X.copy(),columns=chanel)
    quantiles = df.quantile(0.2)
    for col in chanel:
        if col in df.columns:
            df[col] = df[col].apply(normalize_value, q=quantiles[col])
        else:
            df[col] = np.nan
    df[chanel] = df[chanel].apply(zscore)
    adata.X = np.array(df)
    return df

def read_file_info(directory ,info_dir,chanel = None):
    file_list = os.listdir(directory)
    csv_files = [file for file in file_list if file.endswith('.csv')]
    dfs = []
    if info_dir.endswith('.csv'):
        info = pd.read_csv(info_dir)
    if info_dir.endswith('.xlsx'):
        info = pd.read_excel(info_dir)
    info['id'] = info['id'].astype(str)
    # 逐个读取CSV文件，并将其存储为DataFrame对象
    for file in csv_files:
        if file.split('.')[0] not in list(info['id']):
            continue
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        
        df.columns = df.columns.str.replace(': Cell: Mean','')
        df.columns = df.columns.str.replace('Centroid X µm','x')
        df.columns = df.columns.str.replace('Centroid Y µm','y')
        df.columns =  [i.split('(')[0] for i in df.columns]
        df.columns = df.columns.str.replace(' ','')
        df = normal(df,chanel)
        df['p_id'] = file.split('.')[0]
        dfs.append(df)
        combined_df = pd.concat(dfs, ignore_index=True)
    data_table = combined_df.drop(['x', 'y','p_id'],axis=1)
    adata = AnnData(data_table)
    # 设置行和列的名称（如果有需要）
    adata.obs_names = data_table.index  # 行名称，即细胞名
    adata.var_names = data_table.columns  # 列名称，即基因名

    for i in info.columns[1:]:
        tmp = {}
        for n in range(info.shape[0]):
            id = str(info['id'][n]).split('.')[0]
            tmp[id] = info[i][n]
        adata.obs[i] = adata.obs['p_id'].map(tmp)

    return adata,info