# ---------------------------------------------------------------------------- #
#                      Authored by Matheus Ferreira Silva                      #
#                           github.com/MatheusFS-dev                           #
# ---------------------------------------------------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ---------------------- Generate synthetic data ---------------------- #
n_samples = 300
n_features = 2
n_clusters = 2
random_state = 42

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, 
                  cluster_std=1.5, random_state=random_state)

# ---------------------- Fit KMeans clustering model ---------------------- #
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
kmeans.fit(X)

# ---------------------- Create meshgrid for decision boundary ---------------------- #
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict labels for each point in the meshgrid
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ---------------------- Set up LaTeX rendering and font settings ---------------------- #
plt.rcParams.update({
    'text.usetex': True,  # Use LaTeX for text rendering
    'font.family': 'serif',  # Use serif fonts
    'font.serif': 'Times New Roman',  # Set Times New Roman as the serif font
})

# ---------------------- Plot decision boundaries ---------------------- #
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='jet')

# ---------------------- Plot the data points ---------------------- #
scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=30, cmap='jet', edgecolors='k')

# ---------------------- Add a legend with LaTeX ---------------------- #
legend = plt.legend(
    scatter.legend_elements()[0],
    ['No Illness', 'Illness'],
    loc='upper right',
    bbox_to_anchor=(1, 1),
    fontsize=12,
    frameon=True,
    title=r'\textbf{Arterial Hypertension}',
    title_fontsize='13'
)

# Adjust layout to prevent cutoff
plt.tight_layout()
plt.subplots_adjust(right=0.85)  # Leaves space for the legend

# ---------------------- Customize the legend frame ---------------------- #
frame = legend.get_frame()
frame.set_edgecolor('black')  # Set the edge color
frame.set_linewidth(1)  # Set the line width

# ---------------------- Show the plot ---------------------- #
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering with Decision Boundary")
plt.show()
