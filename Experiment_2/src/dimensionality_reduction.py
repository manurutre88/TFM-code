import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import json
import umap
import os

BASE_DIR = os.getcwd()
json_path = os.path.join(BASE_DIR,'Experiment_2','config','config.json')
umap_path = os.path.join(BASE_DIR,'Experiment_2','results','graphs','umap_data')

with open(json_path) as file:
    config = json.load(file)

# ////////////////////////////////////////////////////////////////////////////////////////////

def umap_data(df, n_neighbors, min_dist, n_components,sufix=''):

    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    X_umap = umap_model.fit_transform(df)

    if config.get('graficar_umap', 1):
        fig = plt.figure()
        if n_components == 2:
            plt.scatter(X_umap[:, 0], X_umap[:, 1], s=20)
            plt.xlabel('UMAP component 1')
            plt.ylabel('UMAP component 2')
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], s=10)
            ax.set_label('UMAP component 1')
            ax.set_label('UMAP component 2')
            ax.set_zlabel('UMAP component 3')
        plt.title(f'UMAP\nn_neigh={n_neighbors}, min_dist={min_dist}')
        #plt.savefig(umap_path+f"/umap_data_{n_neighbors}_{min_dist}_{n_components}.png")
        #plt.savefig(umap_path+f"/umap_data_{n_neighbors}_{min_dist}_{n_components}.pdf")

    return X_umap

# ////////////////////////////////////////////////////////////////////////////////////////////
