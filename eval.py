import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_var_cor(x, ax=None, ret=False, *args, **kwargs):
    if type(x) == pd.DataFrame:
        corr = x.corr().values
    else:
        corr = np.corrcoef(x, rowvar=False)
    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    if type(ax) is None:
        f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, *args, **kwargs)
    if ret:
        return corr

def plot_corr_diff(y: np.ndarray, y_hat: np.ndarray, plot_diff=False, *args, **kwargs):
    fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    y_corr = plot_var_cor(y, ax=ax[0], ret=True)
    y_hat_corr = plot_var_cor(y_hat, ax=ax[1], ret=True)

    if plot_diff:
        diff = abs(y_corr - y_hat_corr)
        sns.set(style="white")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(diff, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(diff, ax=ax[2], mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, *args, **kwargs)
    for i, label in enumerate(['y', 'y_hat', 'diff']):
        ax[i].set_title(label)
        ax[i].set_yticklabels(y.columns.values)
        ax[i].set_xticks(list(np.arange(0.5,26.5,1)))
        ax[i].set_xticklabels(y.columns.values, rotation='vertical')
        plt.tight_layout()

        

def eucl_corr(y, y_hat):
    from torch import Tensor
    if type(y) == Tensor:
        y = pd.DataFrame(y.data.numpy())
        y_hat = pd.DataFrame(y_hat.data.numpy())
    return matrix_distance_euclidian(y.corr().fillna(0).values, y_hat.corr().fillna(0).values)


def matrix_distance_abs(ma, mb):
    return np.sum(np.abs(np.subtract(ma, mb)))


def matrix_distance_euclidian(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))


def wasserstein_distance(y, y_hat):
    return stats.wasserstein_distance(y, y_hat)


def get_duplicates(real_data, synthetic_data):
    df = pd.merge(real_data, synthetic_data.set_index('trans_amount'), indicator=True, how='outer')
    duplicates = df[df._merge == 'both']
    return len(duplicates), duplicates

def plot_dim_red(df, how='PCA', cont_names=None, cat_names=None):
    from sklearn.decomposition import PCA
    import seaborn as sns
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    if cat_names:
        df = df.drop(cat_names, axis=1)

    cont_names = df.columns if not cont_names else cont_names
    x = pca.fit_transform(df[cont_names])
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(ax=ax, x=x[:, 0], y=x[:, 1])

def plot_stats(real_dict, fakes):
    if not isinstance(fakes, list):
        fakes = [fakes]
    for fake_dict in fakes:
        scaler = MinMaxScaler()
        real = scaler.fit_transform(real_dict['num'].values)
        fake = scaler.transform(fake_dict['num'].values)
        means_x = np.mean(real, axis=0)
        means_y = np.mean(fake, axis=0)    
        std_x = np.std(real, axis=0)
        std_y = np.std(fake, axis=0)
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        sns.scatterplot(x=means_x, y=means_y, ax=ax[0])
        sns.scatterplot(x=std_x, y=std_y, ax=ax[1])
        plt.show()