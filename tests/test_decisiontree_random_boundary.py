import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

np.random.seed(45)

# number of points for learning (small)
n_points = 30
# number of points for testing the training results (big)
N_points = 100

# compute poistions on square grid
X = np.meshgrid(np.linspace(0,1,n_points),np.linspace(0,1,n_points))
X = np.array(X).reshape(2,-1).T

# remove upper triangle
cond = X[:,0] > X[:,1]
X = X[cond]

# simulate random shadow
Y = np.random.rand(n_points, n_points)
Y = gaussian_filter(Y, sigma=n_points/5)
Y = Y > 0.5
Y = Y[np.triu_indices(n_points, k=1)]
Y = Y.astype(int)

Y = np.random.randint(0,2,size=len(X))

# learn a decision tree
tree = DecisionTreeClassifier(max_depth=15)
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






