import numpy as np
import copy
import random

from sklearn.metrics import silhouette_score


class KMeans:
    def __init__(self, max_cluster_value: float = 10.0):
        self.max_cluster_value = max_cluster_value

    def clustering(self, data, k: int):
        self.k = k
        self.n = len(data)
        self.dim = len(data[0])

        cluster = [[0 for i in range(self.dim)] for q in range(self.k)]

        for i in range(self.dim):
            for q in range(self.k):
                cluster[q][i] = random.randint(0, int(self.max_cluster_value))

        cluster_content = self.data_distribution(data, cluster)

        previous_cluster = copy.deepcopy(cluster)
        while True:
            cluster = self.cluster_update(cluster, cluster_content)
            cluster_content = self.data_distribution(data, cluster)
            if cluster == previous_cluster:
                break

            previous_cluster = copy.deepcopy(cluster)

        return cluster_content

    def cluster_update(self, cluster, cluster_content):
        k = len(cluster)
        for i in range(k):
            for q in range(self.dim):
                updated_parameter = 0
                for j in range(len(cluster_content[i])):
                    updated_parameter += cluster_content[i][j][q]

                if len(cluster_content[i]) != 0:
                    updated_parameter = updated_parameter / len(cluster_content[i])

                cluster[i][q] = updated_parameter

        return cluster

    def data_distribution(self, data, cluster):
        cluster_content = [[] for i in range(self.k)]

        for i in range(self.n):
            min_distance = float('inf')
            suitable_cluster = -1
            for j in range(self.k):
                distance = 0
                for q in range(self.dim):
                    distance += (data[i][q] - cluster[j][q]) ** 2

                distance = distance ** (1 / 2)
                if distance <= min_distance:
                    min_distance = distance
                    suitable_cluster = j

            cluster_content[suitable_cluster].append(data[i])

        return cluster_content

    def silhouette_score(self, data, cluster_labels):
        return silhouette_score(data, cluster_labels)
