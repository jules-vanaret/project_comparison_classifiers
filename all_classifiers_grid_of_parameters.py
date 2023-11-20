import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from time import time
from functools import partial

np.random.seed(45)

classifiers = [
    partial(SVC, kernel='linear'),
    partial(SVC, kernel='rbf'),
    DecisionTreeClassifier
]

names = ['linSVM', 'kSVM (RBF kernel)', 'DecisionTree']

classifier_params = [
    {'C': np.logspace(-2,2,3)},
    {'C': np.logspace(-2,2,3), 'gamma': np.logspace(-2,2,3)},
    {'max_depth': np.arange(1,10,3), 'min_samples_leaf': np.arange(1,10,3)}
]


# number of points for learning (small)
n_points = 30
# number of points for testing the training results (big)
N_points = 100
# number of time to repeat operations to compute average time
n_repeats = 100

# compute poistions on square grid
X = np.meshgrid(np.linspace(0,1,n_points),np.linspace(0,1,n_points))
X = np.array(X).reshape(2,-1).T

# remove upper triangle
cond = X[:,0] > X[:,1]
X = X[cond]

# initialize labels (all to 1=light)
Y = np.random.rand(n_points, n_points)
Y = gaussian_filter(Y, sigma=n_points/5)
Y = Y > 0.5
Y = Y[np.triu_indices(n_points, k=1)]
Y = Y.astype(int)

# create a finer grid for testing the results 
X_big = np.meshgrid(np.linspace(0,1,N_points),np.linspace(0,1,N_points))
X_big = np.array(X_big).reshape(2,-1).T
cond = X_big[:,0] > X_big[:,1]
X_big = X_big[cond]



for i,(name, classifier, params) in enumerate(zip(names, classifiers, classifier_params)):
    print(name)
    
    if len(params.values()) == 1:
        subplots_size = (1, len(list(params.values())[0]))
        figsize = (len(list(params.values())[0])*1.5, 1.5)
    else:
        subplots_size = (len(list(params.values())[0]), len(list(params.values())[1]))
        figsize = (len(list(params.values())[1])*1.5, len(list(params.values())[0])*1.5)
    
    print(figsize)
    fig, axes = plt.subplots(*subplots_size, figsize=figsize)

    # axes[0].scatter(X[:,0],X[:,1],c=Y, s=2)
    # axes[0].set_title('training set')


    # learn a decision tree
    if len(params.values()) == 1:
        for ind_param,param in enumerate(list(params.values())[0]):
            # initialize classifier
            classifier_func = classifier(**{list(params.keys())[0]:param})
            # fit the classifier
            classifier_func.fit(X,Y)
            # predict the labels for the finer grid
            Y_big = classifier_func.predict(X_big)
            # plot the results
            axes[ind_param].scatter(X_big[:,0],X_big[:,1],c=Y_big, s=0.1)
            axes[ind_param].set_title(
                f'{list(params.keys())[0]}={param}',
                fontsize=7
            )
            axes[ind_param].set_xticks([])
            axes[ind_param].set_yticks([])

    else:
        for i,param1 in enumerate(list(params.values())[0]):
            for j,param2 in enumerate(list(params.values())[1]):
                # initialize classifier
                classifier_func = classifier(**{list(params.keys())[0]:param1, list(params.keys())[1]:param2})
                # fit the classifier
                classifier_func.fit(X,Y)
                # predict the labels for the finer grid
                Y_big = classifier_func.predict(X_big)
                # plot the results
                axes[i,j].scatter(X_big[:,0],X_big[:,1],c=Y_big, s=0.1)

                if name == 'DecisionTree':
                    axes[i,j].set_title(
                        f'{list(params.keys())[0]}={param1}, \n{list(params.keys())[1]}={param2}',
                        fontsize=7
                    )
                else:
                    axes[i,j].set_title(
                        f'{list(params.keys())[0]}={param1}, {list(params.keys())[1]}={param2}',
                        fontsize=7
                    )
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])


    fig.tight_layout()
plt.show()



    


