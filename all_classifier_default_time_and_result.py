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
    partial(DecisionTreeClassifier, max_depth=5)
]

names = ['linSVM', 'kSVM (RBF kernel)', 'DecisionTree']


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

fig, axes = plt.subplots(1,len(classifiers)+1, figsize=(10,3))

axes[0].scatter(X[:,0],X[:,1],c=Y, s=2)
axes[0].set_title('training set')

for i,(name, classifier) in enumerate(zip(names, classifiers)):
    print(name)

    t0= time()
    # learn a decision tree
    for _ in range(n_repeats):
        classifier_func = classifier()
        classifier_func.fit(X,Y)
    t1 = time()
    print('\taverage fitting time:', (t1-t0)/n_repeats)

    for _ in range(n_repeats):
        Y_big = classifier_func.predict(X_big)
    t2 = time()
    print('\taverage predicting time:', (t2-t1)/n_repeats)

    # plot the results
    axes[i+1].scatter(X_big[:,0],X_big[:,1],c=Y_big, s=0.1)

    axes[i+1].set_title(name)


fig.tight_layout()
plt.show()



    


