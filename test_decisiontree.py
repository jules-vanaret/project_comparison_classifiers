import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt


# number of points for learning (small)
n_points = 10
# number of points for testing the training results (big)
N_points = 100

# compute poistions on square grid
X = np.meshgrid(np.linspace(0,1,n_points),np.linspace(0,1,n_points))
X = np.array(X).reshape(2,-1).T

# remove upper triangle
cond = X[:,0] > X[:,1]
X = X[cond]

# initialize labels (all to 1=light)
Y = 1*np.ones(shape=(len(X),1), dtype=int)

# add shadow (0) to some points
shadow_cond = np.logical_and(
    X[:,0] > 0.5,
    X[:,0] <0.9
)
shadow_cond = np.logical_and(
    shadow_cond,
    X[:,1] < 0.3
)
Y[shadow_cond] = 0

# learn a decision tree
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X,Y)

# create a finer grid for testing the results 
X_big = np.meshgrid(np.linspace(0,1,N_points),np.linspace(0,1,N_points))
X_big = np.array(X_big).reshape(2,-1).T
cond = X_big[:,0] > X_big[:,1]
X_big = X_big[cond]

Y_big = tree.predict(X_big)

# plot the results
fig, axes = plt.subplots(1,2)
axes[0].scatter(X[:,0],X[:,1],c=Y)
axes[1].scatter(X_big[:,0],X_big[:,1],c=Y_big)
plt.show()

print(export_text(tree, feature_names=['x','y']))






