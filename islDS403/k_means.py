import numpy as np
from tqdm.auto import tqdm

def ed(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans():

    def __init__(self, k=5, max_iter=100, plot_step=False):
        self.k = k
        self.max_iter = max_iter
        self.plot_step = plot_step

        self.clusters = [[] for _ in range(self.k)]
        self.centeroids = []


    def fit(self, x):
        self.x = x
        self.n, self.d = x.shape

        samples = np.random.choice(self.n, self.k, replace=False)
        self.centeroids = [self.x[i] for i in samples]

        # for _ in range(self.max_iter):
        for _ in tqdm(range(self.max_iter), desc='Kmeans clustering fitting'):
            self.clusters = self._get_clusters(self.centeroids)
            old_centeroids = self.centeroids
            self.centeroids = self._get_centeroids(self.clusters)

            # if self.plot_step:
            #     self.plot()

            if self._is_converged(old_centeroids, self.centeroids):
                # print("Centroids are achieved")
                break
        
        return self._get_clusterlabels(self.clusters)

        
    def _get_clusterlabels(self, clusters):
        labels = np.empty(self.n)
        for cluster_idx, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cluster_idx
        return labels

    def _get_clusters(self, centeroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            new_center = self._closest_center(sample, centeroids)
            clusters[new_center].append(idx)
        return clusters
    
    def _closest_center(self, sample, centeroids):
        distance = [ed(sample, point) for point in centeroids]
        return np.argmin(distance)

    def _get_centeroids(self, clusters):
        # centeroids = np.zeros(self.k, self.n)
        centeroids = []
        for cluster_idx, cluster in enumerate(clusters):
            centeroids.append(np.mean(self.x[cluster], axis=0))
        return np.array(centeroids)
    
    def _is_converged(self, old_centeroids, centeroids):
        distance = [ed(old_centeroids[i], centeroids[i]) for i in range(self.k)]
        return sum(distance) == 0