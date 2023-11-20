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

You can install them through [pip](https://pip.pypa.io/en/stable/getting-started/) with the following command:
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