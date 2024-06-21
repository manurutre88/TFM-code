import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import json
import umap
import os

BASE_DIR = os.getcwd()
json_path = os.path.join(BASE_DIR,'Experiment_1','config','config.json')
tsne_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','tsne_data')
umap_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','umap_data')

with open(json_path) as file:
    config = json.load(file)

# ////////////////////////////////////////////////////////////////////////////////////////////

def tsne_data(df, perplexity, n_components, n_iter=10000):
    tsne = TSNE(perplexity=perplexity, n_components=n_components, n_iter=n_iter)
    X_tsne = tsne.fit_transform(df)

    if config.get('graficar_tsne', 1):
        fig = plt.figure()
        if n_components == 2:
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=20)
            plt.xlabel('t-SNE component 1')
            plt.ylabel('t-SNE component 2')
        elif n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2])
            ax.set_xlabel('t-SNE component 1')
            ax.set_ylabel('t-SNE component 2')
            ax.set_zlabel('t-SNE component 3')
        plt.title(f't-SNE\nperp={perplexity}, n_iter= {n_iter}')
        plt.savefig(tsne_path+f"/tsne_data_{perplexity}_{n_components}_{n_iter}.png")
        plt.savefig(tsne_path+f"/tsne_data_{perplexity}_{n_components}_{n_iter}.pdf")

    return X_tsne

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
        plt.savefig(umap_path+f"/umap_data_{n_neighbors}_{min_dist}_{n_components}.png")
        plt.savefig(umap_path+f"/umap_data_{n_neighbors}_{min_dist}_{n_components}.pdf")

    return X_umap

# ////////////////////////////////////////////////////////////////////////////////////////////
