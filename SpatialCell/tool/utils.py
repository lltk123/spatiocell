import pandas as pd

def sorted_df(df, sort_col, group_col = None,**kwargs):
    """
    Sort a DataFrame by a specified column or by the mean value of a column within groups.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be sorted.
    sort_col : str
        The column name to sort by.
    group_col : str, optional
        The column name for grouping. If provided, the function sorts the DataFrame 
        based on the mean value of `sort_col` within each group defined by `group_col`.
        Default is None.
    **kwargs : dict
        Additional keyword arguments to be passed to `sort_values`.

    Returns
    -------
    pandas.DataFrame
        The sorted DataFrame.

    Notes
    -----
    - If `group_col` is None, the DataFrame is sorted directly by `sort_col`.
    - If `group_col` is provided, the DataFrame is first grouped by `group_col`, 
      the mean of `sort_col` is calculated for each group, and the groups are sorted 
      based on these mean values. The resulting DataFrame is then sorted by the 
      order of the groups.

    """
    if group_col is None:
        df = df.sort_values(by=sort_col,**kwargs)
        return df
    else:
        mean_values = df.groupby(group_col)[sort_col].mean().sort_values(**kwargs).index
        df[group_col] = pd.Categorical(df[group_col], categories=list(mean_values), ordered=True)
        df = df.sort_values(by=group_col)
        return df