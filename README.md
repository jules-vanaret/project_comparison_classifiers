# Comparison of binary classifiers for the estimation of the boundary of a shadow cast on a triangle


## Table of contents

* [Installation](#Installation)
* [Content](#Content)

## Installation
To use the scripts, please install the following required Python packages:
* numpy
* matplotlib
* scikit-learn
* scipy
* tqdm

You can install them through [pip](https://pypi.org/project/pip/) with the following command:
```bash
pip install numpy matplotlib scikit-learn scipy tqdm
```

## Content
This repository contains the following files:
* 'decisiontree_on_arbitrary_boundary.py': Example script to show how to generate a random arbitrary boundary and use the decision tree classifier to approxime its boundary. The script also prints the splitting constraints hierarchy.
* 'display_many_simulated_shadows.py': Example script to show how to generate many random arbitrary boundaries and display them.
* 'all_classifiers_default_time_and_result.py': Script to compare the default inference time of the classifiers. The script plots the resulting boundary learnt for comparison.
* 'all_classifiers_f1_vs_nrays.py': Script to compare the F1 score of all classifiers for different number of rays used in the training set. The F1 score is averaged over many random shadows.
* 'all_classifiers_(inference/training)time_vs_nrays.py: Script to compare the inference/training time of all classifiers for different number of rays used in the training set. The time is averaged over many random shadows.
* 'all_classifiers_grid_of_parameters.py': Script to show the qualitative impact of the most important parameters of each classifier on the resulting boundary. The script plots the resulting boundary learnt for comparison.





















This repository is composed of :

* [TO BE DONE]

## Table of contents

* [Installation](#Installation)
* [TODO](#TODO)
* [DONE](#DONE)

## Installation

Basically, the most important non-standard package is [napari](https://napari.org/). If it is already setup on your machine, most of this package should work, and you can skip to the installation of the package (last line of this part).
If not, I provide instructions to install a conda environment compatible with the code.

To create the conda environment, use :

```bash
conda env create -f conda_packajules.yml
```

Or update env :

```bash
conda env update -n env-packajules -f conda_packajules.yml --prune
```

Then get into env-packajules conda env :

```bash
conda env activate env-packajules
```

Finally, make sure local python package is installed :

```bash
# from repo root (the DOT "." is important !!!)
cd packajules;pip install -e .
```


## TODO

* add things to do

## DONE

* created a README 