import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import json
import os

BASE_DIR = os.getcwd()
json_path = os.path.join(BASE_DIR,'Experiment_2','config','config.json')
dbscan_path = os.path.join(BASE_DIR,'Experiment_2','results','graphs','dbscan_data')

with open(json_path) as file:
    config = json.load(file)

def dbscan_data(X, epsilon, min_samples, metodo):
    model_DBSCAN = DBSCAN(eps=epsilon, min_samples=min_samples)
    model_DBSCAN.fit(X)
    labels = model_DBSCAN.labels_

    print(f'Grupos detectados: {len(set(labels)) - (1 if -1 in labels else 0)}')
    print(f'Número de datos clasificados como ruído: {list(labels).count(-1)}')

    if config.get('graficar_dbscan', 1):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis', alpha=0.5)
        plt.legend(*scatter.legend_elements(), title='Groups')
        plt.title(f'Clustering DBSCAN - epsilon={epsilon} - min_samples={min_samples}')
        plt.xlabel(f'{metodo} component 1')
        plt.ylabel(f'{metodo} component 2')
        plt.savefig(dbscan_path+f"/DBSCAN_epsilon_{epsilon}_min_samples_{min_samples}_{metodo}.png")
        plt.savefig(dbscan_path+f"/DBSCAN_epsilon_{epsilon}_min_samples_{min_samples}_{metodo}.pdf")

    return labels

#////////////////////////////////////////////////////////////////////////////////////////////