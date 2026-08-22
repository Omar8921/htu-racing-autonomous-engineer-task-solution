import numpy as np
from sklearn.cluster import DBSCAN

def cluster_points(
    points: np.ndarray,
    eps: float,
    min_samples: int
):
    # Handle empty input
    if points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Initialize model
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    # Fit model and predict clusters
    labels = dbscan.fit_predict(points)

    # Get cluster IDs and remove noise (-1)
    unique = np.unique(labels)
    unique = unique[unique != -1]

    clusters = []

    # Find centroid of each cluster
    for cluster_id in unique:
        cluster = points[labels == cluster_id]

        cx, cy, cz = np.mean(cluster, axis=0)

        clusters.append((cx, cy, cz))

    return np.array(clusters, dtype=np.float32)