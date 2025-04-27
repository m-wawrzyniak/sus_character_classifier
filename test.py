import data_handling as dh
import dimensionality_reduction as dr
import clustering_models as cm

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

random.seed(121)

def test_loading_images():
    input_ex = 'C:/Users/matti/PycharmProjects/sus_character_classifier/input_files/subsamp50.csv'
    images_df = dh.load_images(input_ex)

    print(images_df.iloc[0]['img'].shape)  # its an array, thats how you access the array

    plt.imshow(images_df.iloc[0]['img'], cmap='gray')
    plt.show()  # thats how you show them


def test_sampling_dataset(n_img):
    train_dir = "C:/Users/matti/PycharmProjects/sus_character_classifier/training_samples"
    save_dir = f"C:/Users/matti/PycharmProjects/sus_character_classifier/input_files/subsamp{n_img}.csv"
    ss_input_dir = dh.create_subsample_input_file(train_dir, save_dir, n_img=n_img)

def test_spliting_complete():
    train_dir = "C:/Users/matti/PycharmProjects/sus_character_classifier/training_samples"
    save_dir = 'C:/Users/matti/PycharmProjects/sus_character_classifier/input_files'
    dh.create_complete_input_file(train_dir, save_dir, 0.2)


def test_pca_choice():
    input_ex = 'C:/Users/matti/PycharmProjects/sus_character_classifier/input_files/subsamp1024.csv'
    images_df = dh.load_images(input_ex)

    X_reduced, pca_obj = dr.reduce_pca(images_df, 512)
    n_comps = dr.plot_explained_var(pca_obj, show_plot=True)
    print(n_comps)


def test_hac():
    """
    There are 26 letters in english alphabet.
    Using ward linkage at dist_thresh = 60.0 we got ~27 clusters.
    :return:
    """
    input_ex = 'C:/Users/matti/PycharmProjects/sus_character_classifier/input_files/subsamp256.csv'
    images_df = dh.load_images(input_ex)

    _, pca_obj = dr.reduce_pca(images_df, 256)
    n_comps = dr.plot_explained_var(pca_obj, show_plot=False)

    X_reduced, _ = dr.reduce_pca(images_df, n_comps)

    labels = cm.perform_hac(X_reduced, thresh=60.0)
    print(f"Number of clusters: {len(np.unique(labels))}")
    #cm.plot_dendrogram(X_reduced)

    return images_df, labels


def test_saving_output(img_df, labels):
    save_path_csv = "C:/Users/matti/PycharmProjects/sus_character_classifier/output_files/result256.csv"
    save_path_html = "C:/Users/matti/PycharmProjects/sus_character_classifier/output_files/result256.html"
    dh.save_output_csv(img_df, labels, save_path_csv)
    dh.save_output_html(img_df, labels, save_path_html)


def test_main_hac():
    input_ex = 'C:/Users/matti/PycharmProjects/sus_character_classifier/input_files/tr_input.csv'
    save_path_csv = "C:/Users/matti/PycharmProjects/sus_character_classifier/output_files/tr_result.csv"
    save_path_html = "C:/Users/matti/PycharmProjects/sus_character_classifier/output_files/tr_result.html"
    images_df = dh.load_images(input_ex)

    _, pca_obj = dr.reduce_pca(images_df, n_comp=1024)
    n_cluster_opt = dr.plot_explained_var(pca_obj, show_plot=False)
    print(f'PCA components for 95% var: {n_cluster_opt}')

    X_reduced, _ = dr.reduce_pca(images_df, n_cluster_opt)
    labels = cm.perform_hac(X_reduced, thresh=200.0)
    print(f"Number of clusters: {len(np.unique(labels))}")

    dh.save_output_csv(images_df, labels, save_path_csv)
    dh.save_output_html(images_df, labels, save_path_html)

"""dataset_path = "C:/Users/matti/data/training_samples"
input_file_path = "C:/Users/matti/data//complete_input.csv"
dh.create_complete_input_file(dataset_path, input_file_path)"""
