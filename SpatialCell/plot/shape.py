import seaborn as sns
import matplotlib.pyplot as plt

def boxplot(adata,uns_name,shape_name):
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(figsize=(7, 6))
    df = adata.uns[uns_name]
    sns.boxplot(
        df, x=shape_name, y="Source",
        whis=[0, 100], width=.6, palette="vlag"
    )
    sns.stripplot(df, x=shape_name, y="Source", size=4, color=".3")
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    return f,ax