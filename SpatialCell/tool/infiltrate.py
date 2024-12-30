import numpy as np
import pandas as pd
import sys
sys.path.append('F:/HLL/SpatialCell')
from SpatialCell.tool import morphological
from scipy.ndimage import distance_transform_edt

def com_dis(mask,distances,points,distances_r):
    for point in points:
        if point[0]>=mask.shape[1]:
            point[0] = mask.shape[1]-1
        if point[1]>=mask.shape[0]:
            point[1] = mask.shape[0]-1
        if mask[point[1], point[0]] == 1:
            distances[point[1], point[0]] = distances_r[point[1], point[0]]*-1
    nearest_distances = distances[points[:, 1], points[:, 0]]
    return nearest_distances


def dis(pointa,mask):
    pointa = pointa.astype(np.int32)
    distances = distance_transform_edt(1-mask, sampling=[1, 1])
    distances_r = distance_transform_edt(mask, sampling=[1, 1])
    dis = com_dis(mask,distances,pointa,distances_r)
    return dis

def infltrate(adata ,
              label = 'label',
              id='Image',
              batch='Parent',
              center_cell = 'SPON2',
              x = 'Centroid X µm',
              y='Centroid Y µm'):
    obs_name = f'infiltrate_to_{center_cell}'
    adata.obs[obs_name]= np.nan
    for i in adata.obs[id].unique():
        sub0 = adata[adata.obs[id] == i]
        if batch is None:
            xmin,ymin = int(min(sub0.obs[x])),int(min(sub0.obs[y]))
            xmax,ymax = int(max(sub0.obs[x])),int(max(sub0.obs[y]))
            # xlim = xmax - xmin
            # ylim = ymax - ymin
            center = sub0[adata.obs[label] == center_cell]

            mask_point = np.array(center.obs[[x,y]])
            mask_point = mask_point-np.array((xmin,ymin))
            fitter_point = morphological.filtered(mask_point, 0.95)  # Filter points
            if fitter_point.shape[0] == mask_point.shape[0]:
                print(f'Image:{i} Parent:{j} not have filtered points')
            _, mask, a, _, _, _, _ = morphological.com(fitter_point, 0.03)  # Compute metrics
            if a == 0: 
                print(f'Image:{i} Parent:{j} not have a shape')
            else:
                for target in sub0.obs[label].unique():
                    pointa = np.array(sub0.obs[[x,y]][sub0.obs[label] == target])
                    pointa = pointa-np.array((xmin,ymin))
                    dis_ = dis(pointa,mask)
                    adata.obs.loc[sub0.obs.index[sub0.obs[label] == target],obs_name] = dis_
        else:
            for j in sub0.obs[batch].unique():
                sub1 = sub0[sub0.obs[batch] == j]
                xmin,ymin = int(min(sub1.obs[x])),int(min(sub1.obs[y]))
                xmax,ymax = int(max(sub1.obs[x])),int(max(sub1.obs[y]))
                # xlim = xmax - xmin
                # ylim = ymax - ymin
                center = sub1[sub1.obs[label] == center_cell]
                mask_point = np.array(center.obs[[x,y]])
                mask_point = mask_point-np.array((xmin,ymin))
                fitter_point = morphological.filtered(mask_point, 0.95)  # Filter points
                if fitter_point.shape[0] == mask_point.shape[0]:
                    print(f'Image:{i} Parent:{j} not have filtered points')
                _, mask, a, _, _, _, _ = morphological.com(fitter_point, 0.03)  # Compute metrics
                if a == 0: 
                    print(f'Image:{i} Parent:{j} not have a shape')
                else:
                    for target in sub1.obs[label].unique():
                        pointa = np.array(sub1.obs[[x,y]][sub1.obs[label] == target])
                        pointa = pointa-np.array((xmin,ymin))
                        dis_ = dis(pointa,mask)
                        adata.obs.loc[sub1.obs.index[sub1.obs[label] == target],obs_name] = dis_