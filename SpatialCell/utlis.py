import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def split_dict_by_intervals(data_dict, intervals):
    """
    将字典根据键值对中的值划分为多个区间。

    Args:
        data_dict (dict): 输入字典，格式为 {key: value}
        intervals (list): 定义区间边界，例如 [0, 25, 50, 75, 100]

    Returns:
        dict: 包含每个区间的键的列表，区间作为键
    """
    # 创建一个字典存储结果
    result = {f"{intervals[i]}-{intervals[i+1]}": [] for i in range(len(intervals) - 1)}
    
    # 遍历字典并按值划分区间
    for key, value in data_dict.items():
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i+1]:  # 值落在当前区间
                result[f"{intervals[i]}-{intervals[i+1]}"].append(key)
                break
    
    return result


def filter_non_positive_correlation(X, Y, threshold=0.1):
    """
    筛选不符合正相关关系的数据点
    :param X: 自变量 (list or np.array)
    :param Y: 因变量 (list or np.array)
    :param threshold: 去掉某点后相关性提高的最小值
    :return: 符合正相关的数据点索引，和被筛选掉的索引
    """
    X = np.array(X)
    Y = np.array(Y)
    valid_indices = []
    outliers = []

    # 计算整体相关系数
    overall_r, _ = pearsonr(X, Y)
    if overall_r <= 0:
        raise ValueError("数据整体不满足正相关性，无法筛选。")

    # 遍历每个点，计算去掉该点后的相关系数
    for i in range(len(X)):
        X_new = np.delete(X, i)
        Y_new = np.delete(Y, i)
        r, _ = pearsonr(X_new, Y_new)

        # 如果去掉该点后相关系数显著提高，则认为该点不符合正相关
        if r - overall_r > threshold:
            outliers.append(i)
        else:
            valid_indices.append(i)

    return valid_indices, outliers