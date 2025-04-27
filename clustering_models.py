from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np


def perform_hac(X_prep, thresh, linkage='ward'):
    """
    Performs Hierarchical Agglomerative Clustering (HAC) without predefining number of clusters.

    Args:
        X_pca (np.ndarray): PCA-reduced data.
        thresh (float): The linkage distance threshold to cut the tree.
        linkage (str): Linkage criterion ('ward', 'complete', 'average', 'single').

    Returns:
        labels (np.ndarray): Cluster labels for each sample.
    """
    hac = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=thresh,
        linkage=linkage
    )
    labels = hac.fit_predict(X_prep)
    return labels


def decide_threshold(size):
    """
    I don't know whether this is sketchy or not, but in order to keep the generalization
    of the model for different sizes of the input dataset I noticed that the threshold
    has to be adjusted.
    When tried on subset sizes of: 256, 512, 1024, 2048, 4096 and full set
    this is the linear regression obtained for best threshold.
    :param size:
    :return:
    """
    return 60 + 0.0203*(size-256)


def plot_dendrogram(X, method='ward', sample_size=256):
    """
    Plots a dendrogram for a subsample of data.

    Args:
        X (np.ndarray): Input data.
        method (str): Linkage method.
        sample_size (int): How many samples to plot.
    """
    np.random.seed(121)
    idx = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    linked = sch.linkage(X_sample, method=method)

    plt.figure(figsize=(12, 6))
    sch.dendrogram(linked)
    plt.title('Dendrogram (sample)')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()