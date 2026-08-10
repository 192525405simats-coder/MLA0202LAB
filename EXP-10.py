from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

X, y = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.60,
    random_state=42
)

model = GaussianMixture(
    n_components=3,
    random_state=42
)

model.fit(X)

labels = model.predict(X)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis"
)

plt.title("Expectation Maximization Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()