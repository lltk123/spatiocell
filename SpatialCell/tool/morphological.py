import numpy as np
from scipy.spatial import distance
import pandas as pd
import math
import cv2
import alphashape
from skimage.morphology import skeletonize
import sknw
import numpy as np
import networkx as nx
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

def _longest_path_from_node(graph, u):
    visited = {i: False for i in list(graph.nodes)}
    distance = {i: -1 for i in list(graph.nodes)}
    idx2node = dict(enumerate(graph.nodes))

    try:
        adj_lil = nx.to_scipy_sparse_matrix(graph, format="lil")
    except AttributeError:
        adj_lil = nx.to_scipy_sparse_array(graph, format="lil")
    adj = {i: [idx2node[neigh] for neigh in neighs] for i, neighs in zip(graph.nodes, adj_lil.rows)}
    weight = nx.get_edge_attributes(graph, "weight")

    distance[u] = 0
    queue = deque()
    queue.append(u)
    visited[u] = True
    while queue:
        front = queue.popleft()
        for i in adj[front]:
            if not visited[i]:
                visited[i] = True
                source, target = min(i, front), max(i, front)
                distance[i] = distance[front] + weight[(source, target)]
                queue.append(i)

    farthest_node = max(distance, key=distance.get)

    longest_path_length = distance[farthest_node]
    return farthest_node, longest_path_length
def _longest_path_length(graph):
    # first DFS to find one end point of longest path
    node, _ = _longest_path_from_node(graph, list(graph.nodes)[0])
    # second DFS to find the actual longest path
    _, longest_path_length = _longest_path_from_node(graph, node)
    return longest_path_length

def dbscan_with_kdtree(full_data, kdtree, distance_threshold=99):
    """
    使用子集的 DBSCAN 结果结合 KD 树在完整数据中标记并删除噪声点。
    
    参数：
    - full_data: ndarray，完整数据集，形状 (n_samples, n_features)
    - sample_size: int，从完整数据中随机抽取的子集大小
    - dbscan_eps: float，DBSCAN 的 eps 参数
    - dbscan_min_samples: int，DBSCAN 的 min_samples 参数
    - distance_threshold: float，距离 KD 树的最大距离，超出该距离的点将被标记为噪声
    
    返回：
    - cleaned_data: ndarray，移除噪声后的数据集
    - noise_indices: ndarray，噪声点的索引
    """
    distances, _ = kdtree.query(full_data, k=5)
    # 1. 计算每行的平均值
    average_distances = distances.mean(axis=1)

    # 2. 计算 95% 分位数阈值
    threshold = np.percentile(average_distances, distance_threshold)

    noise_mask = average_distances  < threshold
    clean_indices = np.where(noise_mask)[0]

    # 移除噪声点
    cleaned_data = full_data[clean_indices]

    return cleaned_data, clean_indices
def grid_search_dbscan(data, eps_range, min_samples_range,target = 0.95):
    best_params = (eps_range[0], min_samples_range[0])
    min_sub = 1e5
    result = np.array([-1]*data.shape[0])
    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
            labels = db.labels_
            noise_ratio = sum(labels == -1) / len(labels)  # 计算噪声比例
            if noise_ratio==0:
                continue
            if abs(np.log2( noise_ratio / (1-target) )) < min_sub:  # 达到目标比例
                best_params = (eps, min_samples)
                min_sub = abs(np.log2( noise_ratio / (1-target) ))
                result = labels
                if (noise_ratio < target+0.01 )& (noise_ratio > target-0.01):
                    return best_params,result
    return best_params,result

def filtered(full_data,proportion = 0.95):
    subs = 10000
    if full_data.shape[0] < subs:
        subs = full_data.shape[0]
    randix = np.random.choice(full_data.shape[0],subs,replace = False)
    sub_adata = full_data[randix]
    eps_range = np.linspace(100, 300, 10)
    min_samples_range = range(5, 20)
    best_params,labels = grid_search_dbscan(sub_adata, eps_range, min_samples_range,target = proportion)
    data = sub_adata 
    db = DBSCAN(eps=best_params[0], min_samples=best_params[1]).fit(data)
    labels = db.labels_  # -1 表示噪声点
    filtered_data = data[labels != -1]  # 删除噪声点
    if filtered_data.shape[0] == 0:
        return full_data
    kdtree = KDTree(filtered_data)
    cleaned_data, _ = dbscan_with_kdtree(full_data,kdtree,distance_threshold=95)
    return cleaned_data


def com(point, k=0.03):
    """
    Computes morphological features of a shape, including area, perimeter, 
    curl, elongation, and linearity, based on an input set of points.

    Parameters
    ----------
    point : array-like
        The input point set representing the shape (e.g., a collection of 2D coordinates).
    k : float, optional
        The alpha shape parameter controlling the level of detail of the generated shape. 
        Default is 0.03.

    Returns
    -------
    masks : array-like
        A binary mask array representing the filled alpha shape.
    a : float
        Area of the alpha shape.
    p : float
        Perimeter of the alpha shape.
    curl : float
        Curliness of the shape, defined as the deviation of the shape's perimeter 
        from its longer dimension.
    elongation : float
        Elongation of the shape, defined as the ratio of the shorter to the longer dimension 
        of its bounding rectangle.
    linearity : float
        Linearity of the skeletonized shape, defined as the ratio of the longest 
        path in the skeleton to the total skeleton length.

    Process
    -------
    1. Compute the alpha shape for the input point set using the `alphashape` library.
    2. Calculate basic metrics:
       - Area (`a`) and perimeter (`p`) of the alpha shape.
       - Minimum rotated rectangle of the alpha shape.
    3. Derive advanced metrics:
       - Curliness (`curl`) based on the ratio of the longer dimension to the 
         fiber approximation of the shape.
       - Elongation (`elongation`) based on the ratio of the shorter to the longer dimension 
         of the bounding rectangle.
    4. Generate a binary mask of the alpha shape and skeletonize it.
    5. Construct a graph from the skeletonized mask and calculate linearity 
       based on the longest path and total path lengths.

    Notes
    -----
    - If the area of the alpha shape is zero, the function returns default zeros 
      for all metrics.
    - Handles both simple and multi-part shapes (multi-polygons).

    """
    # Generate alpha shape using the specified alpha parameter
    alpha_shape = alphashape.alphashape(point, alpha=k)
    a = alpha_shape.area  # Area of the alpha shape
    if a == 0:
        return alpha_shape, 0, 0, 0, 0, 0, 0

    # Calculate perimeter and minimum rotated rectangle
    p = alpha_shape.length
    min_rect = alpha_shape.minimum_rotated_rectangle

    # Calculate "fiber" metric to approximate the shape's flexibility
    fiber = 4 * a / (p - math.sqrt(p * p - 16 * a))

    # Compute distances in the bounding rectangle to derive curliness and elongation
    dist_matrix = distance.cdist(
        list(min_rect.exterior.coords), 
        list(min_rect.exterior.coords), 
        'euclidean'
    )
    longer = max(pd.DataFrame(dist_matrix).loc[1, 2], pd.DataFrame(dist_matrix).loc[2, 3])
    curl = 1 - longer / fiber

    shorter = min(pd.DataFrame(dist_matrix).loc[1, 2], pd.DataFrame(dist_matrix).loc[2, 3])
    elongation = 1 - shorter / longer

    # Extract bounds and initialize binary mask for alpha shape
    xlim = alpha_shape.bounds[2] - alpha_shape.bounds[0]
    ylim = alpha_shape.bounds[3] - alpha_shape.bounds[1]
    masks = np.zeros((int(ylim), int(xlim)), dtype=np.uint8)

    # Fill the binary mask based on the alpha shape geometry
    try:
        for polygon in alpha_shape.geoms:
            x = pd.DataFrame(list(polygon.exterior.coords))
            x = x - [alpha_shape.bounds[0], alpha_shape.bounds[1]]
            x = np.array(x, np.int32)
            roiMask01 = np.zeros((int(ylim), int(xlim)), dtype=np.uint8)
            roiMask01 = cv2.fillPoly(roiMask01, [x], (1))
            masks = np.logical_or(masks, roiMask01)
    except:
        x = pd.DataFrame(list(alpha_shape.exterior.coords))
        x = x - [alpha_shape.bounds[0], alpha_shape.bounds[1]]
        x = np.array(x, np.int32)
        roiMask01 = np.zeros((int(ylim), int(xlim)), dtype=np.uint8)
        roiMask01 = cv2.fillPoly(roiMask01, [x], (1))
        masks = roiMask01

    # Skeletonize the binary mask and build a graph
    ske = skeletonize(masks).astype(np.uint16)
    graph = sknw.build_sknw(ske)

    # Compute longest path and total length of skeleton
    longest_path_length = _longest_path_length(graph)
    total_length = sum(nx.get_edge_attributes(graph, 'weight').values())
    linearity = longest_path_length / total_length

    return alpha_shape,masks, a, p, curl, elongation, linearity


def shape(adata, 
          target_cell, 
          label='label', 
          x='Centroid X µm', 
          y='Centroid Y µm', 
          id='Image', 
          batch=None, 
          name=None):
    """
    Calculate shape metrics for specific cells in spatial data.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial data with coordinates and labels.
    target_cell : str
        The target cell type or category to analyze, specified in the `label` column of `adata.obs`.
    label : str, optional
        The column in `adata.obs` that contains the cell type or category labels. Default is 'label'.
    x : str, optional
        The column name for the x-coordinate of the cell centroids. Default is 'Centroid X µm'.
    y : str, optional
        The column name for the y-coordinate of the cell centroids. Default is 'Centroid Y µm'.
    id : str, optional
        The column in `adata.obs` that contains unique identifiers for different images or regions. Default is 'Image'.
    batch : str, optional
        If provided, this specifies a column in `adata.obs` for grouping cells within each image. Default is None.
    name : str, optional
        The name to save the resulting DataFrame in `adata.uns`. If None, a default name is generated using the `target_cell`.

    Returns
    -------
    None
        The function updates the `adata.uns` attribute with the calculated shape metrics DataFrame.
    
    Process
    -------
    1. Filter the data for the target cell type specified by `target_cell`.
    2. Iterate through each unique image (or region) specified in the `id` column.
    3. Calculate shape metrics (area, perimeter, curl, elongation, linearity) for all cells within each image or batch.
    4. Save the resulting metrics as a DataFrame in `adata.uns`, optionally grouping by a batch column.

    Notes
    -----
    - Shape metrics are calculated using the `com` and `filtered` functions, which should handle point filtering and metric computation.
    - If no valid shape can be computed for an image or batch, warnings are printed.
    - The resulting DataFrame contains the calculated shape metrics and includes a "Source" column indicating the image or batch.

    """
    sub0 = adata[adata.obs[label] == target_cell]
    result = {}
    
    # Iterate through unique images or regions
    for i in sub0.obs[id].unique():
        sub1 = sub0[sub0.obs[id] == i]
        sub1_result = {}
        
        if batch is None:  # No batch specified
            point = np.array(sub1.obs[[x, y]])
            fitter_point = filtered(point, 0.95)  # Filter points
            if fitter_point.shape[0] == point.shape[0]:
                print(f'Image:{i} not have filtered points')
            _, _, a, p, curl, elongation, linearity = com(fitter_point, 0.03)  # Compute metrics
            if a == 0:
                print(f'Image:{i} not have a shape')
            shape_list = [a, p, curl, elongation, linearity]
            sub1_result['All'] = shape_list
        
        else:  # Batch specified
            for j in sub1.obs[batch].unique():
                sub2 = sub1[sub1.obs[batch] == j]
                point = np.array(sub2.obs[[x, y]])
                fitter_point = filtered(point, 0.95)  # Filter points
                if fitter_point.shape[0] == point.shape[0]:
                    print(f'Image:{i} Parent:{j} not have filtered points')
                _, _, a, p, curl, elongation, linearity = com(fitter_point, 0.03)  # Compute metrics
                if a == 0:
                    print(f'Image:{i} Parent:{j} not have a shape')
                shape_list = [a, p, curl, elongation, linearity]
                sub1_result[j] = shape_list
        
        # Convert metrics to a DataFrame
        result[i] = pd.DataFrame(
            sub1_result,
            index=['Area', 'Perimeter', 'Curl', 'Elongation', 'Linearity']
        ).T
    
    # Combine results into a single DataFrame
    combined_df = pd.concat(
        [df.assign(Source=key) for key, df in result.items()],
        ignore_index=False
    )
    combined_df['ROI'] = combined_df.index  # Add Region of Interest column
    
    # Save the resulting DataFrame in adata.uns
    if name is None:
        adata.uns[f'shape_{target_cell}'] = combined_df
    else:
        adata.uns[name] = combined_df
