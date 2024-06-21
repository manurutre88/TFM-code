import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json 
import os

#/////////////////////////////////////////////////////////////////////////////////////////

from src.data_processing import import_data,group_data
from src.feature_extraction import ext_caract
from src.dimensionality_reduction import tsne_data,umap_data
from src.clustering import kmeans_data,agrupar,dbscan_data,elbow_kmeans,silhouette_kmeans,kdist_dbscan
from src.plotting import boxplot_metricas,boxplot_grupos,graph_data


#/////////////////////////////////////////////////////////////////////////////////////////

BASE_DIR = os.getcwd()
json_path = os.path.join(BASE_DIR,'Experiment_1','config','config.json')

with open(json_path) as file:
    config = json.load(file)

#/////////////////////////////////////////////////////////////////////////////////////////

print('\n////////// DATA IMPORT ///////////\n')
param_import_mode = config.get("modo_import")
### IMPORTACION E TRANSFORMACIÓN DE DATOS
df = import_data(modo=param_import_mode)
print(df)
print(df.max().max())
print(df.min().min())

if config.get('graph_dataset', 1):
    graph_data(df.T)

## Agrupación por diferentes unidades temporais
df_weekly, df_monthly = group_data(df)
print(df_weekly)
print(df_monthly)

#/////////////////////////////////////////////////////////////////////////////////////////

print('\n////////// FEATURE EXTRACTION ///////////\n')

list_caract = []
lista_df = [df]
sufix = ['_D']
for i,j in zip(lista_df, sufix):
    list_caract.append(ext_caract(i,j))

df_caract = pd.DataFrame()
for k in list_caract:
    df_caract = pd.concat([df_caract,k], axis=1)
print(df_caract)
metricas = df_caract.columns

list_caract = []
lista_df = [df, df_weekly, df_monthly]
sufix = ['_D', '_W', '_M']
for i,j in zip(lista_df, sufix):
    list_caract.append(ext_caract(i,j))

df_caract_ext = pd.DataFrame()
for k in list_caract:
    df_caract_ext = pd.concat([df_caract_ext,k], axis=1)
print(df_caract_ext)
metricas_ext = df_caract_ext.columns

#/////////////////////////////////////////////////////////////////////////////////////////

### DIMENSIONAL REDUCTION
print('\n////////// DIMENSIONAL REDUCTION ///////////\n')

# tSNE
if config.get('barrido_tsne', 1):
    print("\n//////// TSNE //////////\n")
    perplexity_list = [3,4,5,6]
    n_components_list = [2]
    n_iter_list =[10000]
    for i in perplexity_list:
        for j in n_components_list:
            for k in n_iter_list:
                _ = tsne_data(df_caract_ext.loc[:,'mean_D':'var_D'], perplexity=i, n_components=j, n_iter=k)
    if config.get('graficar_tsne', 1):
        plt.show()

# UMAP
if config.get('barrido_umap', 1): 
    print("\n/////// UMAP /////////\n")               
    n_neighbors_list = [4,5,6,7]
    min_dist_list = [0.01,0.05,0.1,0.2]
    n_components_list = [2]
    for i in n_neighbors_list:
        for j in min_dist_list:
            for k in n_components_list:
                _ = umap_data(df_caract_ext.loc[:,'mean_D':'var_D'], n_neighbors=i, min_dist=j, n_components=k)
    if config.get('graficar_umap', 1):
        plt.show()

# Recollemos os datos no espazo reducido para as mellores combinacións
tsne_output = tsne_data(df_caract_ext.loc[:,'mean_D':'var_D'], perplexity=5, n_components=2, n_iter=10000)
umap_output = umap_data(df_caract_ext.loc[:,'mean_D':'var_D'], n_neighbors=6, min_dist=0.01, n_components=2)

if (config.get('graficar_tsne', 1) or config.get('graficar_umap', 1)): plt.show()

#/////////////////////////////////////////////////////////////////////////////////////////

### CLUSTERING
print('\n////////// CLUSTERING ///////////\n')

## K-MEANS
if config.get('execute_kmeans', 1):

    print("\n////////////// K-MEANS /////////////////\n")

    if config.get('graficar_elbow', 1): 
        elbow_kmeans(df_caract_ext.loc[:,'mean_D':'var_D'])

    if config.get('graficar_silhouette', 1):
        silhouette_kmeans(df_caract_ext.loc[:,'mean_D':'var_D'])

    n_clusters = [5]
    for n in n_clusters:
        df_caract_ext[f"labels_kmeans_tsne_{n}"] = kmeans_data(tsne_output,n_clusters=n,metodo='tsne')
        df_caract_ext[f"labels_kmeans_umap_{n}"] = kmeans_data(umap_output,n_clusters=n,metodo='umap')
    
    if config.get('graficar_kmeans', 1): plt.show()

    print(df_caract_ext)

    grupos_kmeans_D = agrupar(df_caract_ext,'kmeans','_D')

    for metrica in metricas:
        boxplot_metricas(grupos_kmeans_D, 'GRUPOS_kmeans_umap_5', metrica, 'boxplot_users_kmeans_umap_5')

    boxplot_grupos(df,grupos_kmeans_D['GRUPOS_kmeans_umap_5'], 'boxplot_users_kmeans_umap_5')

## DBSCAN
if config.get('execute_dbscan', 1):

    print("\n////////////// DBSCAN /////////////////\n")

    if config.get('graficar_boxplots', 1):
        kdist_dbscan(umap_output)

    df_caract_ext["labels_dbscan_umap"] = dbscan_data(umap_output, epsilon=1.5, min_samples=4, metodo='umap')
    df_caract_ext["labels_dbscan_tsne"] = dbscan_data(tsne_output, epsilon=4, min_samples=4, metodo='dbscan')

    if config.get('graficar_dbscan', 1): plt.show()

    grupos_dbscan_1 = agrupar(df_caract_ext, 'dbscan', '_D')

    for metrica in metricas:
        boxplot_metricas(grupos_dbscan_1, 'GRUPOS_dbscan_umap', metrica, 'boxplot_users_dbscan_umap')

    boxplot_grupos(df,grupos_dbscan_1['GRUPOS_dbscan_umap'],'boxplot_users_dbscan_umap')


#/////////////////////////////////////////////////////////////////////////////////////////

df.to_excel(BASE_DIR+'/Experiment_1/results/DataFrames/df.xlsx')
#df_weekly.to_excel(BASE_DIR+'/Experiment_1/results/DataFrames/df_weekly.xlsx')
#df_monthly.to_excel(BASE_DIR+'/Experiment_1/results/DataFrames/df_monthly.xlsx')
df_caract.to_excel(BASE_DIR+'/Experiment_1/results/DataFrames/df_caract.xlsx')
df_caract_ext.to_excel(BASE_DIR+'/Experiment_1/results/DataFrames/df_caract_ext.xlsx')
print('\n\n\n')









