import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

np.random.seed(45)



# number of points for learning (small)
n_points = 100
# number of shadows to plot
n_shadows=5

# compute positions on square grid
X = np.meshgrid(np.linspace(0,1,n_points),np.linspace(0,1,n_points))
X = np.array(X).reshape(2,-1).T

# remove upper triangle
cond = X[:,0] > X[:,1]
X = X[cond]


fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(12, 4))


for i in range(3):
    for j in range(8):
        Y = np.random.rand(n_points, n_points)
        Y = gaussian_filter(Y, sigma=n_points/10)
        Y = Y > 0.5
        Y = Y[np.triu_indices(n_points, k=1)]
        Y = Y.astype(int)

        axes[i,j].scatter(X[:,0], X[:,1], c=Y, s=2)
        axes[i,j].xaxis.set_visible(False)
        axes[i,j].yaxis.set_visible(False)


fig.tight_layout()
plt.show()


    


