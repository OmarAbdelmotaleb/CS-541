#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# Clustering Demographics
#####################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv("income.csv")
df["sex"] = df["sex"].map(lambda s: 1 if "Female" in s else 0 ) # 0 if " Male" else 1

races = {}
races[" White"] = 10
races[" Black"] = 11
races[" Asian-Pac-Islander"] = 12
races[" Amer-Indian-Eskimo"] = 13
races[" Other"] = 14

df["race"] = df["race"].map(lambda s: races[s])

rship = {}
rship[" Husband"] = 20
rship[" Not-in-family"] = 21
rship[" Other-relative"] = 22
rship[" Own-child"] = 23
rship[" Unmarried"] = 24
rship[" Wife"] = 25

df["relationship"] = df["relationship"].map(lambda s: rship[s])

# df.to_csv("income_mod.csv")
# Adapted from https://www.youtube.com/watch?v=W4fSRHeafMo&ab_channel=AladdinPersson
class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100
        self.plot_figure = True
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]

    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

        return centroids

    def create_clusters(self, X, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.K)]

        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            )
            clusters[closest_centroid].append(point_idx)

        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid

        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def plot_fig(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    def fit(self, X):
        centroids = self.initialize_random_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)

            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)

            diff = centroids - previous_centroids

            if not diff.any():
                print("Termination criterion satisfied")
                break

        # Get label predictions
        y_pred = self.predict_cluster(clusters, X)

        if self.plot_figure:
            self.plot_fig(X, y_pred)

        return y_pred

sex = np.array(df["sex"])
race = np.array(df["race"])
relationship = np.array(df["relationship"])

sex_race = np.column_stack((sex,race))
sex_relationship = np.column_stack((sex,relationship))
race_relationship = np.column_stack((race,relationship))

X = np.vstack((sex_race, sex_relationship))
X = np.vstack((X,race_relationship))

Kmeans = KMeansClustering(X, 2)
y_pred = Kmeans.fit(X)

