import numpy as np
import pandas as pd
def compute_density(
    adata,
    group_name,
    labels=None,
    batch=None,
    cell_type="label",
    spatial_x="Centroid X µm",
    spatial_y="Centroid Y µm",
    bins=20,
):
    """
    Calculate the 2D density distribution of specified cell types within spatial coordinates,
    grouped by a specified category and optionally by batch.

    Parameters:
    -----------
    adata : AnnData
        An AnnData object containing cell observations and spatial coordinates.
    group_name : str
        Column name used for grouping (e.g., sample ID or experimental group).
    labels : list, optional
        List of cell types to calculate density for. If None, all cell types in `cell_type` will be used.
    batch : str, optional
        Column name for further sub-grouping (e.g., ROI regions). If None, no sub-grouping is applied.
    cell_type : str, default='label'
        Column name indicating cell type labels.
    spatial_x : str, default='Centroid X µm'
        Column name for the X-coordinate in spatial data.
    spatial_y : str, default='Centroid Y µm'
        Column name for the Y-coordinate in spatial data.
    bins : int, default=20
        Number of bins for the 2D histogram in both X and Y directions.

    Returns:
    --------
    None
        The results are stored in `adata.uns['density_2d']` as a DataFrame with 
        density values and metadata (sample ID and ROI information).

    Notes:
    ------
    - Each cell type's density is computed using a 2D histogram within the spatial coordinate range.
    - Results are aggregated and aligned with sample IDs and ROIs for downstream analysis.

    """
    result = {}  # Dictionary to store density data for each target label
    id_list, roi_list = [], []  # Lists to store sample IDs and ROI information

    # Use all unique cell types if no specific labels are provided
    if labels is None:
        labels = adata.obs[cell_type].unique()

    # Iterate through unique groups defined by `group_name`
    for group_id in adata.obs[group_name].unique():
        group_data = adata[adata.obs[group_name] == group_id]  # Subset data for the current group

        # Iterate through subgroups defined by `batch`, or process the entire group if `batch` is None
        batch_values = group_data.obs[batch].unique() if batch else [None]
        for batch_id in batch_values:
            if batch:
                batch_data = group_data[group_data.obs[batch] == batch_id]  # Subset data for the current batch
            else:
                batch_data = group_data

            # Extract spatial coordinates
            coords = batch_data.obs[[spatial_x, spatial_y]].values
            x, y = coords[:, 0], coords[:, 1]
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()

            # Calculate density for each specified cell type
            for target in labels:
                target_data = batch_data[batch_data.obs[cell_type] == target]  # Subset data for the target label
                target_coords = target_data.obs[[spatial_x, spatial_y]].values

                # Compute density using a 2D histogram
                target_x, target_y = target_coords[:, 0], target_coords[:, 1]
                H, _, _ = np.histogram2d(
                    target_x, target_y, bins=bins, range=[[xmin, xmax], [ymin, ymax]]
                )
                H_flat = list(H.flatten())  # Flatten the 2D density grid into a 1D list

                # Accumulate density data for the current target label
                if target not in result:
                    result[target] = H_flat
                else:
                    result[target] += H_flat

            # Append metadata for sample ID and batch/ROI
            id_list += [group_id] * len(H_flat)
            roi_list += [batch_id] * len(H_flat)

    # Convert the results dictionary into a DataFrame and add metadata columns
    density_df = pd.DataFrame(result)
    density_df["id"] = id_list
    density_df["roi"] = roi_list

    # Store the density results in the AnnData object
    adata.uns["density_2d"] = density_df
