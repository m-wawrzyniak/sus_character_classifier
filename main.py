import data_handling as dh
import dimensionality_reduction as dr
import clustering_models as cm

import numpy as np
import random
import os
import sys

random.seed(121)

# Paths
root = os.path.dirname(os.path.abspath(__file__))
save_csv_path = root + "/output_files/complete_results.csv"
save_html_path = root + "/output_files/complete_results.html"

# User input of the input_file.csv
if len(sys.argv) < 2:
    print("Please provide the input dataset path:")
    sys.exit(1)
input_dir = sys.argv[1]
if not os.path.isfile(input_dir):
    print(f"Provided path '{input_dir}' is not a valid file.")
    exit(1)
print(f"Dataset path: {input_dir}")

# Loading and preprocessing the data
X_df = dh.load_images(input_dir)
X_size = X_df.shape[0]

# Parametrizing the PCA and reducing the dimensionality of the data
_, pca_obj = dr.reduce_pca(X_df, n_comp=1024)
n_comp_opt = dr.plot_explained_var(pca_obj, show_plot=False)
print(f'PCA components for 95% var: {n_comp_opt}')
X_reduced, _ = dr.reduce_pca(X_df, n_comp_opt)

# Performing HAC on the dataset
dist_thresh = cm.decide_threshold(X_size)
labels = cm.perform_hac(X_reduced, thresh=dist_thresh)
print(f"Number of clusters: {len(np.unique(labels))}")

# Saving the results
dh.save_output_csv(X_df, labels, save_csv_path)
dh.save_output_html(X_df, labels, save_html_path)
