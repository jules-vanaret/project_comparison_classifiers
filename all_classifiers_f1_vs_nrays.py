import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from time import time
from functools import partial
from sklearn.metrics import f1_score, jaccard_score
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map

np.random.seed(45)

classifiers = [
    partial(SVC, kernel='linear'),
    partial(SVC, kernel='rbf'),
    DecisionTreeClassifier
]

names = ['linSVM', 'kSVM (RBF kernel)', 'DecisionTree']


# number of points for learning (small)
n_points = 100
# number of shadows of which to average the metrics
n_shadows=100
# maximum decimation factor (1 means no decimation)
max_decimation_factor = 100
decimation_factors = np.arange(1, max_decimation_factor, 3)

# compute positions on square grid
X = np.meshgrid(np.linspace(0,1,n_points),np.linspace(0,1,n_points))
X = np.array(X).reshape(2,-1).T

# remove upper triangle
cond = X[:,0] > X[:,1]
X = X[cond]

n_rays = len(X)/decimation_factors

def parallel_func(foo, n_points, classifiers, names, decimation_factors, X):
    # initialize labels (all to 1=light)
    Y = np.random.rand(n_points, n_points)
    Y = gaussian_filter(Y, sigma=n_points/10)
    Y = Y > 0.5
    Y = Y[np.triu_indices(n_points, k=1)]
    Y = Y.astype(int)

    all_scores_par = np.zeros(shape=(len(classifiers), len(decimation_factors)))

    
    for index_dec_factor,decimation_factor in enumerate(decimation_factors):
        X_train = X[::decimation_factor]
        Y_train = Y[::decimation_factor]

        for index_classifier,(name, classifier) in enumerate(zip(names, classifiers)):

            classifier_func = classifier()
            classifier_func.fit(X_train,Y_train)

            Y_fit = classifier_func.predict(X)

            all_scores_par[index_classifier,index_dec_factor] += f1_score(Y, Y_fit)

    return all_scores_par

func =  partial(
    parallel_func, 
    n_points=n_points, 
    classifiers=classifiers, 
    names=names, 
    decimation_factors=decimation_factors, 
    X=X
)

all_scores = process_map(func, range(n_shadows), max_workers=7, chunksize=6)
all_scores = np.mean(all_scores, axis=0)


fig, axes = plt.subplots(1,len(classifiers), figsize=(9,3))
for i, name in enumerate(names):
    axes[i].plot(n_rays, all_scores[i], '-o')
    axes[i].set_xscale('log')
    axes[i].set_title(name)
    axes[i].set_xlabel('number of rays')
    axes[i].set_ylabel('f1 score')
    axes[i].set_ylim(0.3,1)


fig.tight_layout()


plt.show()


    


