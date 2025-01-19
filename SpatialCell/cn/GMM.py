import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def aic_bic(data , extent = (1,10)):
    random_state = 42
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

    return aic_values,bic_values

def aicbic_pl(data ,extent = (1,10)):
    aic_values,bic_values = aic_bic(data , extent = extent)
    components = np.arange(extent[0], extent[1], 1)
    plt.figure(figsize=(8, 5))
    plt.plot(components, aic_values, label='AIC', marker='o')
    plt.plot(components, bic_values, label='BIC', marker='s')
    plt.xticks(components)
    plt.title("AIC and BIC for Gaussian Mixture Models")
    plt.xlabel("Number of Components")
    plt.ylabel("Information Criterion Value")
    plt.legend()
    plt.grid()