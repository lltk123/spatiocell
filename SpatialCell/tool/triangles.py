# -*- coding: utf-8 -*-
# @Author: HLL
# @Date: 2024-12-30
# @Description: This file contains the implementation of triangulation algorithms.
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

# 计算每个三角形的边长
def calculate_edge_lengths(triangles, points):
    edges = []
    for triangle in triangles:
        # 获取三角形的三个顶点坐标
        p1, p2, p3 = points[triangle]
        # 计算三条边的长度
        edge_lengths = [
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p2 - p3),
            np.linalg.norm(p3 - p1)
        ]
        edges.append(edge_lengths)
    return np.array(edges)

# 计算每个三角形的面积（使用海伦公式）
def calculate_triangle_area(triangle, points):
    p1, p2, p3 = points[triangle]
    # 三个边长
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p3 - p1)
    # 半周长
    s = (a + b + c) / 2
    # 海伦公式计算面积
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

# 获取每个点的最短边长度和最小面积三角形
def calculate_point_properties(triang, points):
    # 获取所有三角形的边长
    edge_lengths = calculate_edge_lengths(triang.simplices, points)
    
    # 获取所有三角形的面积
    areas = np.array([calculate_triangle_area(triangle, points) for triangle in triang.simplices])
    
    # 初始化数组来存储每个点的最短边长度和最小面积
    min_edge_lengths = np.full(len(points), np.inf)
    min_areas = np.full(len(points), np.inf)
    
    # 对于每个三角形，更新每个点的最短边和最小面积
    for i, triangle in enumerate(triang.simplices):
        # 获取当前三角形的边长和面积
        triangle_edges = edge_lengths[i]
        triangle_area = areas[i]
        
        # 对每个点，更新最短边长度和最小面积
        for point_index,egd_index in zip(triangle,[[0,2],[0,1],[1,2]]):
            min_edge_lengths[point_index] = min(min_edge_lengths[point_index], *triangle_edges[egd_index])
            min_areas[point_index] = min(min_areas[point_index], triangle_area)
    
    return min_edge_lengths, min_areas




def triangulation(
        adata,
        label='label',                
        id='Image',                
        batch='Parent',            
        x='Centroid X µm',         
        y='Centroid Y µm'):        
    """
    Perform Delaunay triangulation and calculate shape metrics for specified cells in spatial data.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial data with coordinates and labels.
    label : str, optional
        The column in `adata.obs` that contains the cell type or category labels. Default is 'label'.
    target_cell : str, optional
        The target cell type to be processed, specified in the `label` column of `adata.obs`. Default is 'SPON2'.
    id : str, optional
        The column in `adata.obs` that contains unique identifiers for different images or regions. Default is 'Image'.
    batch : str, optional
        If provided, this specifies a column in `adata.obs` for grouping cells within each image. Default is 'Parent'.
    x : str, optional
        The column name for the x-coordinate of the cell centroids. Default is 'Centroid X µm'.
    y : str, optional
        The column name for the y-coordinate of the cell centroids. Default is 'Centroid Y µm'.

    Returns
    -------
    None
        The function updates `adata.obs` with calculated triangulation areas and edge lengths.

    Process
    -------
    1. Filter the data for the target cell type specified by `target_cell`.
    2. Iterate through each unique image or region (based on `id` column).
    3. For each image or batch, calculate the Delaunay triangulation and compute shape metrics (area, edge length).
    4. Store the results in `adata.obs` for each cell.

    Notes
    -----
    - If fewer than 3 points are available for triangulation, a warning is printed.
    - The results include triangulation areas and edge lengths for each point in `adata.obs`.
    """
    
    # Initialize columns in `adata.obs` to store triangulation results
    adata.obs['triang_area'], adata.obs['triang_edges_length'] = np.nan, np.nan

    # Iterate over all unique target cell types
    for target_cell in adata.obs[label].unique():
        # Filter the data for the current target cell type
        sub0 = adata[adata.obs[label] == target_cell]
        
        # Iterate over unique images or regions (based on 'id' column)
        for i in sub0.obs[id].unique():
            sub1 = sub0[sub0.obs[id] == i]  # Subset for the current image/region
            
            # If no batch is specified
            if batch is None:
                # Extract coordinates and calculate Delaunay triangulation
                points = np.array(sub1.obs[[x, y]])
                triang = Delaunay(points)
                edges, areas = calculate_point_properties(triang, points)
                
                # Store the calculated edges and areas in `adata.obs`
                edges = pd.Series(edges, index=sub1.obs_names)
                areas = pd.Series(areas, index=sub1.obs_names)
                adata.obs.loc[sub1.obs_names, 'triang_area'] = areas
                adata.obs.loc[sub1.obs_names, 'triang_edges_length'] = edges
            
            # If batch information is provided
            else:
                # Iterate over unique batches (based on 'batch' column)
                for j in sub1.obs[batch].unique():
                    sub2 = sub1[sub1.obs[batch] == j]  # Subset for the current batch
                    
                    # If there are fewer than 3 points, triangulation cannot be computed
                    if sub2.shape[0] < 3:
                        print(f'Not enough points to calculate triangulation for {target_cell} in {i} {j}')
                        continue
                    
                    # Extract coordinates and calculate Delaunay triangulation
                    points = np.array(sub2.obs[[x, y]])
                    triang = Delaunay(points)
                    edges, areas = calculate_point_properties(triang, points)
                    
                    # Store the calculated edges and areas in `adata.obs`
                    edges = pd.Series(edges, index=sub2.obs_names)
                    areas = pd.Series(areas, index=sub2.obs_names)
                    adata.obs.loc[sub2.obs_names, 'triang_area'] = areas
                    adata.obs.loc[sub2.obs_names, 'triang_edges_length'] = edges
