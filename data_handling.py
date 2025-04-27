import pandas as pd
import numpy as np
import cv2
import random
import os
import scipy
from pathlib import Path
from collections import defaultdict

def load_images(path, resize=32, center=True):
    """
    Loads the images provided in an .csv formatted according to the task instruction.
    Transforms them to grayscale, and saves as np.arrays.
    Resizes them into one size and centers them.
    Returns pd.Dataframe, 1st column is path, 2nd is image array representation.
    :param resize:
    :param path:
    :param center:
    :return:
    """
    paths_df = pd.read_csv(path, sep=' ', names=['file_path'])
    images = []

    for i, path in enumerate(paths_df['file_path']):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # reading
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)  # resizing
        if center:
            _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  #creating a binary image
            cent_y, cent_x = scipy.ndimage.center_of_mass(bin_img>0)  # center coordinates
            vec_x = int(np.round(resize/2 - cent_x))  # translation vector at x-axis
            vec_y = int(np.round(resize/2 - cent_y))  # translation vector at y-axis
            trans_matrix = np.float32([[1, 0, vec_x], [0, 1, vec_y]])  # translation matrix for the image
            img = cv2.warpAffine(img, trans_matrix, dsize=(resize, resize), borderValue=(255,))  # affine translation
        images.append(img)

    images_df = paths_df.copy()
    images_df['img'] = images

    return images_df


def create_subsample_input_file(input_dir, save_dir, n_img):
    """
    Creates an .csv input file from the images in the 'path' directory.
    The input file is 'n' random paths from the directory.
    :return:
    """

    input_dir = Path(input_dir)  # create a Path object

    all_pngs = [str((input_dir / f).resolve().as_posix()) # save the absolute path within given directory
                for f in os.listdir(input_dir)
                if (input_dir / f).is_file() and f.lower().endswith('.png')] # if the this path points to file and has .png format

    chosen_pngs = random.sample(all_pngs, min(n_img, len(all_pngs)))  # sample the paths at random, choose n_img of paths
    df = pd.DataFrame(chosen_pngs, columns=['_'])  # save them as df, to easy csv conversion
    df.to_csv(save_dir, index=False, header=False)  # drop column and row names

    print(f'Input .csv file at {save_dir} has been created. n_img = {n_img}')
    return save_dir


def create_complete_input_file(input_dir, save_dir):
    input_dir = Path(input_dir)  # create a Path object
    all_pngs = [str((input_dir / f).resolve().as_posix()) # save the absolute path within given directory
                for f in os.listdir(input_dir)
                if (input_dir / f).is_file() and f.lower().endswith('.png')] # if the this path points to file and has .png format
    random.shuffle(all_pngs)  # shuffling the images


    df = pd.DataFrame(all_pngs, columns=['_'])

    df.to_csv(save_dir, index=False, header=False)

    print(f"complete_input.csv saved to {save_dir} ({len(df)})")


def save_output_csv(images_df, labels, save_path):
    """
    Creates and saves the specified .csv format output file.
    :param images_df:
    :param labels:
    :param save_path:
    :return:
    """
    clust_dict = defaultdict(list)
    for fp, label in zip(images_df['file_path'], labels):
        filename = os.path.basename(fp)  # this extracts just the filename
        clust_dict[label].append(filename)  # adds this image to given cluster

    with open(save_path, 'w') as f:  # writing into the file
        for cluster_files in clust_dict.values():
            line = ' '.join(cluster_files)
            f.write(line + '\n')

    print(f'Output .csv saved to {save_path}')


def save_output_html(images_df, labels, output_path):
    """
    Saves the clustering results into .html file, where each cluster
    is separated by a horizontal line.
    """

    # creating the cluster dict
    clust_dict = defaultdict(list)
    for fp, label in zip(images_df['file_path'], labels):
        clust_dict[label].append(fp)  # adds this image to given cluster

    # writing the html
    with open(output_path, 'w') as f:
        f.write('<html><body>\n')
        for cluster in clust_dict.values():
            for img_path in cluster:
                f.write(f'<img src="file:///{img_path}" style="height:48px; margin:5px;">\n')
            f.write('<hr>\n')
        f.write('</body></html>')

    print(f'Output .html saved to {output_path}')

# Legacy code

def create_complete_input_file_legacy(input_dir, save_dir, test_size=0.2):
    input_dir = Path(input_dir)  # create a Path object
    all_pngs = [str((input_dir / f).resolve().as_posix()) # save the absolute path within given directory
                for f in os.listdir(input_dir)
                if (input_dir / f).is_file() and f.lower().endswith('.png')] # if the this path points to file and has .png format
    random.shuffle(all_pngs)  # shuffling the images

    # splitting the dataset
    split_id = int(len(all_pngs) * (1 - test_size))
    tr_paths = all_pngs[:split_id]
    test_paths = all_pngs[split_id:]

    tr_df = pd.DataFrame(tr_paths, columns=['_'])
    test_df = pd.DataFrame(test_paths, columns=['_'])

    # saving the input files into corresponding .csv files
    save_dir = Path(save_dir)
    tr_path = save_dir / 'tr_input.csv'
    test_path = save_dir / 'test_input.csv'
    tr_df.to_csv(tr_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)

    print(f"tr_input saved to {tr_path} ({len(tr_df)})")
    print(f"test_input saved to {test_path} ({len(test_df)})")
