import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import json
import os

BASE_DIR = os.getcwd()
json_path = os.path.join(BASE_DIR,'Experiment_1','config','config.json')
graph_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','graph_data')
metricas_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','boxplot_metricas_grupos')
grupos_path = os.path.join(BASE_DIR,'Experiment_1','results','graphs','boxplot_usuarios_grupos')

with open(json_path) as file:
    config = json.load(file)

# ////////////////////////////////////////////////////////////////////////////////////////////

def graph_data(df_trans):

    title = "Consumo diario por usuario"
    fig = px.line(df_trans, x=df_trans.index, y=df_trans.columns, title=title)
    fig.write_html(graph_path+f"/graph_consumo_anual_diario_users.html")
    fig.show()

    fig = go.Figure()
    for (col_name, col), (_, row)in zip(df_trans.items(),df_trans.iterrows()):
        fig.add_trace(go.Box(y=col, name=col_name))   
    fig.update_layout(title_text="Box Plot de consumo diario de usuarios")
    fig.write_html(graph_path+f"/boxplot_consumo_anual_diario_users.html")
    fig.show()

# ////////////////////////////////////////////////////////////////////////////////////////////

def boxplot_metricas(grupos, metodo_agrupacion, metrica, name):

    if config.get('graficar_boxplots_metricas', 1):
        metrica_grupos = []
        for subgrupo in grupos[metodo_agrupacion].keys():
            metrica_grupos.append([i for i in grupos[metodo_agrupacion][subgrupo][metrica]])

        fig = go.Figure()
        for group, value in zip(grupos[metodo_agrupacion].keys(), metrica_grupos):
            fig.add_trace(go.Box(name=group, y=value))
            
        fig.update_layout(title_text=f"Box Plot {metodo_agrupacion} --> {metrica}")
        fig.write_html(metricas_path+f"/{name}_{metrica}.html")
        fig.show()

# ////////////////////////////////////////////////////////////////////////////////////////////

def boxplot_grupos(df, grupos, name):

    if config.get('graficar_boxplots_grupos', 1):
        for grupo in grupos.keys():
            fig = go.Figure()
            for user in grupos[grupo].index:
                fig.add_trace(go.Box(name=user, y=df.loc[user]))
            fig.update_layout(title_text=f"Box Plot usuarios {grupo}")
            fig.write_html(grupos_path+f"/{name}_{grupo}.html")
            fig.show()

# ////////////////////////////////////////////////////////////////////////////////////////////
