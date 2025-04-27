import data_handling as dh
import dimensionality_reduction as dr
import clustering_models as cm

import numpy as np
import random
import os

random.seed(121)

# Paths
root = os.path.dirname(os.path.abspath(__file__))

# THIS HAS TO BE CHANGED TO TAKE THE PATH OF INPUT LIST FROM THE COMMAND-LINE
dataset_path = "C:/Users/matti/data/training_samples"
input_file_path = root + "/input_files/complete_input.csv"
dh.create_complete_input_file(dataset_path, input_file_path)
# END

save_csv_path = root + "/output_files/complete_results.csv"
save_html_path = root + "/output_files/complete_results.html"

# Loading and preprocessing the data
X_df = dh.load_images(input_file_path)
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
