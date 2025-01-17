# SpatialCell

## 项目简介

SpatialCell 是一个用于分析空间细胞数据的工具包。该工具包提供了一系列函数，用于处理和分析细胞形态学特征、细胞间互作网络、细胞邻域成分等。

## 功能

- **1**: 使用GaussianMixture对细胞进行注释
- **2**: 计算细胞分布形成的形态学特征，包括纤维度、卷曲度和伸长率。
- **3**: 指定细胞的细胞邻域成分分析
- **4**: 计算并可视化细胞类型之间的距离，基于几何空间定位细胞与细胞之间的相互作用网络，搜索特定细胞的空间分布模式
，
## 安装

请确保已安装以下依赖项：

- numpy
- pandas
- scikit-learn
- alphashape
- scipy

你可以使用以下命令安装这些依赖项：

```bash
pip install numpy pandas scikit-learn alphashape scipy
