from sklearn.cluster import SpectralClustering
import numpy as np

clustering = SpectralClustering(n_clusters=2, assign_labels="kmeans", random_state=0)