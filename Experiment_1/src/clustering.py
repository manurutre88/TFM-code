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
json_path = os.path.join(BASE_DIR,'Experiment_1','config','config.json')
silhouette_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','silhouette_kmeans')
elbow_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','elbow_kmeans')
kmeans_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','kmeans_data')
groups_path = os.path.join(BASE_DIR,'Experiment_1','results','groups')
dbscan_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','dbscan_data')

with open(json_path) as file:
    config = json.load(file)

#////////////////////////////////////////////////////////////////////////////////////////////

def silhouette_kmeans(X,max_k=20):
    silhouette_coef = []

    for k in range(2,max_k+1):
        model_kmeans = KMeans(n_clusters=k).fit(X)
        silhouette_coef.append(silhouette_score(X,model_kmeans.fit_predict(X)))

    if config.get('graficar_silhouette', 1):
        plt.figure()
        plt.plot(range(2,max_k+1), silhouette_coef, marker='x')
        plt.title('Coeficiente de silhouette')
        plt.ylabel('Coeficiente')
        plt.xlabel('Número de clusters (K)')
        plt.savefig(silhouette_path+f"/silhouette_kmeans.png")
        plt.show()

#////////////////////////////////////////////////////////////////////////////////////////////

def elbow_kmeans(X,max_k=20):
    inercia = []

    for k in range(1,max_k+1):
        model_kmeans = KMeans(n_clusters=k).fit(X)
        inercia.append(model_kmeans.inertia_)

    if config.get('graficar_elbow', 1):
        plt.figure()
        plt.plot(range(1,max_k+1), inercia, marker='x')
        plt.title('Gráfico de codo (elbow)')
        plt.ylabel('Inercia')
        plt.xlabel('Número de clusters (K)')
        plt.savefig(elbow_path+f"/elbow_kmeans.png")
        plt.show()

#////////////////////////////////////////////////////////////////////////////////////////////

def kmeans_data(X,n_clusters,dim=2,metodo='tsne'):

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    if config.get('graficar_kmeans', 1):
        plt.figure(figsize=(10, 6))
        if dim == 2:
            # Visualizar los clusters encontrados por K-means
            scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis', alpha=0.5)
            plt.legend(*scatter.legend_elements(), title='Grupos')                         
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100, alpha=0.75)
            plt.title(f'Clustering K-means - {metodo} - {n_clusters} clusters')
            plt.xlabel(f'{metodo} component 1')
            plt.ylabel(f'{metodo} component 2')
            plt.savefig(kmeans_path+f"/kmeans_2D__{metodo}_{n_clusters}.png")
            plt.savefig(kmeans_path+f"/kmeans_2D__{metodo}_{n_clusters}.pdf")

    return labels

#////////////////////////////////////////////////////////////////////////////////////////////

def agrupar(df, tecnica, sufix):

    print("//////////////////////////////AGRUPAR/////////////////////////////////")

    grupos = {}
    label_cols = [col for col in df.columns if col.startswith('labels_' + tecnica)]
    #print(label_cols)

    for col in label_cols:

        clave1 = f'GRUPOS_{col[7:]}'
        grupos[clave1] = {}
        print(f"\n\n\n{clave1}\n")
        
        unique_labels = df[col].unique()
        for num in range(len(unique_labels)):
            
            clave2 = f'Grupo_{num}'
            print(f'\n{clave2}')
            print_cols = [col2 for col2 in df if not col2.startswith('labels')]
            print(df.loc[df[col]==num,print_cols])
            grupos[clave1][clave2] = df.loc[df[col]==num,print_cols]
    
    for k1, k1_grupos in grupos.items():

        with pd.ExcelWriter(f'{groups_path}/{k1}_{sufix}.xlsx', engine='xlsxwriter') as writer:
            for k2, k2_df in k1_grupos.items():
                k2_df.to_excel(writer, sheet_name=k2)

    return grupos

#////////////////////////////////////////////////////////////////////////////////////////////

def kdist_dbscan(X,dim=4):

    if config.get('graficar_kdist', 1):
        plt.figure(figsize=(10, 6))
        minPts = 2*dim
        model_KNN = NearestNeighbors(n_neighbors=minPts)
        model_KNN.fit(X)
        distancias, indices = model_KNN.kneighbors(X)

        k_esima_dist = distancias[:,minPts-1]

        k_esima_dist = np.sort(k_esima_dist,)

        if config.get('graficar_kdist', 1):
            plt.figure()
            plt.plot(range(X.shape[0]), k_esima_dist)
            plt.title('4-ésima distancia')
            plt.ylabel('distancia')
            plt.xlabel('Dato')
            plt.show()

#////////////////////////////////////////////////////////////////////////////////////////////

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