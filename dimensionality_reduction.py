from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reduce_pca(X_df, n_comp):
    """
    Flattens the (32,32) np.array to (1024, ) and standardize it using scaler.
    Then applies PCA on the data.
    :param X_df:
    :param n_comp:
    :return:
    """
    X_full = np.stack(X_df['img'].apply(lambda x: x.flatten()).to_numpy())  # shape: (n_samples, 1024)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)  # scaled features

    pca = PCA(n_components=n_comp, random_state=121)  # pca object
    X_pca = pca.fit_transform(X_scaled)  # applying PCA

    return X_pca, pca

def plot_explained_var(pca, target=0.95, show_plot=False):
    """
    Used to settle on the best number of reduced dimensions.
    :param pca:
    :param target: Target explain variance.
    :param show_plot:
    :return:
    """
    exp_var = np.cumsum(pca.explained_variance_ratio_)  # cumulative sum
    target_n_comps = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= target) + 1  # target number of comps

    if show_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(exp_var)+1), exp_var)
        plt.xlabel('n_comps')
        plt.ylabel('cum_exp_var')
        plt.title('cum_exp_var v. n_comps')
        plt.grid(True)
        plt.show()

    return target_n_comps