# Character classifier - SUS Global assignment no.1

## Project description

This project clusters character images into groups based on their visual similarity, without prior knowledge of the classes.
It uses PCA for dimensionality reduction, followed by Hierarchical Agglomerative Clustering (HAC) using Euclidean distance and Ward’s linkage method.

The project generates two types of output (specified at the task instruction):

 - Text file: Lists image filenames grouped by cluster.

 - HTML file: Displays clustered images, separated by horizontal lines for visual clarity.

## Methodology

1. Loading a .csv file formatted just as the task specified.
2. Preprocessing:
   1. Images are loaded as greyscale matrices.
   2. They are centered at center of mass (mass being the value of individual pixels)
   3. Then they are translated so that each image has the center of mass in the same, central pixel
   4. Images are resized to 32x32 size.
   5. The pixel matrices are then flattened.
3. Dimensionality reduction:
   1. Reduction is applied as PCA.
   2. Initially, PCA is applied to high number of dimensions, to finally decide on how many components should be included while retaining 95% of variance.
   3. Final PCA is applied with number of components settled on in the previous step.
4. Clustering:
   1. After some trial and error, I decided that Hierarchical Clustering Algorithm is sufficient and fast-enough to accurately work on this dataset.
   2. I settled on Euclidian distance with ward's linkage, minimizing intra-cluster variance, as different letters should really be similar to one another.
   3. Distance threshold is adaptive relative to the dataset, in order to maintain the best generalization of the model - higher thresholds for larger datasets.
5. Output: Returns both .csv and .html files according to format specified in the task.

## How to run

I tried to make sure that the project is compatible to both Win and Linux.
You can either access the project using Github link or directly from the provided file.

To make it as easy as possible, I create .bat and .sh files, to execute most of the environment
setup. You only need to run in on Win or Linux correspondingly. 

### Running on Win straight from the file
```
cd <path to the project directory called 'sus_character_classifier'>
run_hac.bat
```

### Running on Win from Github
```
git clone https://github.com/m-wawrzyniak/sus_character_classifier
cd sus_character_classifier
run_hac.bat
```

### Running on Linux straight from the file
```
cd <path to the project directory called 'sus_character_classifier'>
run_hac.sh
```

### Running on Linux from Github
```
git clone https://github.com/m-wawrzyniak/sus_character_classifier
cd sus_character_classifier
run_hac.sh
```

After running the executive file and setting up the venv, you will be asked to provide the path to the input .csv file.

All output files will be saved within the project directory in file output_files.

## Author: Mateusz Wawrzyniak, 446 271
