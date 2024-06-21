import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import json 
import os

# Comprobar se, usando como variables de entrada o consumo medio de cada mes, aparecen os clusters orixinais
# Cluster
# 12 variables de entrada --> 12 meses --> consumo acumulado e consumo medio

#/////////////////////////////////////////////////////////////////////////////////////////

from src.dimensionality_reduction import umap_data

#/////////////////////////////////////////////////////////////////////////////////////////

def tuple_2_df(col1,col2):

    data = {'UMAP_x':col1, 'UMAP_y':col2}
    df = pd.DataFrame(data)
    return df

#/////////////////////////////////////////////////////////////////////////////////////////

BASE_DIR = os.getcwd()
json_path = os.path.join(BASE_DIR,'Experiment_2','config','config.json')
cluster_path = os.path.join(BASE_DIR,'data','dataset_auga1','cluster_dataset','cluster_ord_etiquetado.xlsx')
save_path = os.path.join(BASE_DIR,'Experiment_2','results','graphs','avg_month')
graph_path = os.path.join(BASE_DIR,'Experiment_2','results','graphs','graph_data')

with open(json_path) as file:
    config = json.load(file)

#/////////////////////////////////////////////////////////////////////////////////////////

df_label = pd.read_excel(cluster_path, index_col=0).dropna()
#df_label.index = [f'User_{i}' for i in range(1,len(df_label.index)+1)]
print(df_label)
df = df_label.loc[:,df_label.columns!='label']
print(df)
print(df.max().max())

def graph_data(df):

    title = "Consumo diario por usuario"
    fig1 = px.line(df, x=df.index, y=df.columns, title=title)
    fig1.show()

    fig2 = go.Figure()
    for (col_name, col), (_, row)in zip(df.items(),df.iterrows()):
        fig2.add_trace(go.Box(y=col, name=col_name))   
    fig2.update_layout(title_text="Box Plot de consumo diario de usuarios")
    fig2.show()

    return fig1,fig2

fig1,fig2 = graph_data(df.T)
fig1.write_html(graph_path+'/graph_consumo_anual_diario_users.html')
fig2.write_html(graph_path+'/boxplot_consumo_anual_diario_users.html')

### DATA TRANSFORMATION (MONTHLY ACCUMULATED CONSUMPTION AND MEAN)
months = {'mar':3, 'apr':4, 'may':5, 'jun':6 , 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12, 'jan':1, 'feb':2}

df_trans = df.T
df_trans.index = pd.to_datetime(df_trans.index, format='%d-%m-%Y')
'''
df_month_sum = df_trans.resample('M').sum().T
df_month_sum.columns = [m for m in months.keys()]
'''
df_month_mean= df_trans.resample('M').mean().T
df_month_mean.columns = [m for m in months.keys()]

#print(df_month_sum)
print(df_month_mean)

#for neigh in [3,4]:
#    for dist in [0.01,0.1,0.25]:

#for neigh in [3,4]:
#    for dist in [0.01,0.1,0.25]:
for neigh in [3]:
    for dist in [0.25]:
        #umap_out_sum = umap_data(df_month_sum, n_neighbors=neigh, min_dist=dist, n_components=2, sufix='month_sum')
        umap_out_mean = umap_data(df_month_mean, n_neighbors=neigh, min_dist=dist, n_components=2, sufix='month_mean')
        '''
        df_umap_out_sum = tuple_2_df([i[0] for i in umap_out_sum],[i[1] for i in umap_out_sum])
        df_umap_out_sum.index = df_month_sum.index
        df_umap_out_sum['total_mean'] = df_month_sum.mean(axis=1)
        df_umap_out_sum['label'] = df_label['label']
        '''
        df_umap_out_mean = tuple_2_df([i[0] for i in umap_out_mean],[i[1] for i in umap_out_mean])
        df_umap_out_mean.index = df_month_mean.index
        df_umap_out_mean['total_mean'] = df_month_mean.mean(axis=1)
        for m in [m for m in months.keys()]:
            df_umap_out_mean[m] = df_month_mean[m]
        df_umap_out_mean['label'] = df_label['label']
        
        #print(df_umap_out_sum)
        print(df_umap_out_mean)

        def graph_labels(df,col_x,col_y,col_color,name=''):

            df["label"] = df["label"].astype(str)
            fig = px.scatter(df, x=col_x, y=col_y, color=col_color, title=f"{name} // UMAP n_neigh={neigh} min_dist={dist}", hover_data={'index': df.index}, color_discrete_sequence=px.colors.qualitative.Alphabet)
            #fig.write_html(save_path+f'/{name}.html')
            #fig.show()
            return fig
        
        def graph_gradient(df,col_x,col_y,col_color,name=''):

            fig = px.scatter(df, x=col_x, y=col_y, color=col_color, title=f"{name} // UMAP n_neigh={neigh} min_dist={dist}", hover_data={'index': df.index})
            #fig.write_html(save_path+f'/{name}.html')
            fig.show()
            return fig

        #graph_labels(df_umap_out_sum,'UMAP_x','UMAP_y','label','labels_umap_month_sum')
        fig1 = graph_labels(df_umap_out_mean,'UMAP_x','UMAP_y','label','labels_umap_month_mean')
        fig1.write_html(save_path+f'/labels_umap_month_monthly_mean.html')

        #graph_gradient(df_umap_out_sum,'UMAP_x','UMAP_y','total_mean','gradient_umap_month_sum')
        # media dos 12 meses
        fig2 = graph_gradient(df_umap_out_mean,'UMAP_x','UMAP_y','total_mean','gradient_umap_month_monthly_mean')
        fig2.write_html(save_path+f'/gradient_umap_month_monthly_mean.html')
        # media de cada mes
        for m in [m for m in months.keys()]:
            fig = graph_gradient(df_umap_out_mean,'UMAP_x','UMAP_y',m,f'gradient_umap_month_{m}_mean')
            fig.write_html(save_path+f'/gradient_{m}_mean.html')

g_1 = ['User_7','User_32','User_38','User_34','User_48','User_41','User_43','User_11']
g_2 = ['User_27','User_20','User_17','User_24','User_46','User_44','User_47']
g_3 = ['User_5','User_28','User_37','User_36']
g_4 = ['User_45','User_10','User_6','User_8','User_22','User_23','User_13','User_12','User_15','User_49','User_50','User_3','User_51','User_19',
       'User_29','User_40','User_33','User_21','User_9','User_2','User_1',]
g_5 = ['User_31','User_30','User_18','User_25','User_14','User_39','User_26','User_16','User_42','User_4',]

total_means = []
for g in [g_1,g_2,g_3,g_4,g_5]:
    total_means.append(df_umap_out_mean.loc[g,'total_mean'].mean())
print(*total_means)

monthly_means = {m: [] for m in months.keys()}
for m in monthly_means.keys():
    for g in [g_1,g_2,g_3,g_4,g_5]:
        monthly_means[m].append(df_umap_out_mean.loc[g,m].mean())
#print(monthly_means)
#print(df_umap_out_mean.loc[g_1,'mar'].mean())
print(df_umap_out_mean)

def plot_video(df):
    # Crear una columna 'id' para identificar a los usuarios en el gráfico animado
    df['id'] = df.index

    # Derretir el DataFrame para tener una estructura larga
    df_melted = df.melt(id_vars=['UMAP_x', 'UMAP_y', 'label', 'id'], 
                        value_vars=['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan', 'feb'],
                        var_name='month', 
                        value_name='mean_value')
    print(df_melted)
    min_value = 0
    max_value = 700

    # Crear el gráfico de dispersión animado
    fig = px.scatter(df_melted, x='UMAP_x', y='UMAP_y', 
                    animation_frame='month', 
                    animation_group='id',
                    color='mean_value', 
                    hover_name='id', 
                    title='Animated monthly mean consmption value per user',
                    labels={'mean_value': 'Monthly mean'},
                    range_color=[min_value, max_value])

    
    fig.show()
    return fig

fig = plot_video(df_umap_out_mean)
fig.write_html(save_path+'/video_monthly_mean.html')

def assign_group(valor,medias_grupos,etiquetas):
    
    # Asignar grupo en función de valor + cercano
    valor_mas_cercano = min(medias_grupos, key=lambda x: abs(x - valor))
    indice = medias_grupos.index(valor_mas_cercano)
    etiqueta_grupo = etiquetas[indice]
    
    return etiqueta_grupo


etiquetas = [1,2,3,4,5]
group_per_month = pd.DataFrame(index=df_umap_out_mean.index, columns=[m for m in months.keys()])
for user in group_per_month.index:
    for month in group_per_month.columns:
        group_per_month.loc[user,month] = assign_group(df_month_mean.loc[user,month],total_means,etiquetas)
#print(group_per_month)


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////
'''
def sankey_diagram(df):

    # Generar transiciones
    transitions = []
    for i in range(len(df.columns)-1):
        start_month = df.columns[i]
        end_month = df.columns[i + 1]
        transition = df.groupby([start_month, end_month]).size().reset_index(name='count')
        print(transition)
        transition.columns = ['source', 'target', 'value']
        transitions.append(transition)

    # Unir todas las transiciones
    #all_transitions = pd.concat(transitions)
    #print(all_transitions)

    # Crear mapeo de grupos a índices
    #group_to_index = {group: idx for idx, group in enumerate(sorted(df.stack().unique()))}
    #all_transitions['source'] = all_transitions['source'].map(group_to_index)
    #all_transitions['target'] = all_transitions['target'].map(group_to_index)
    #print(group_to_index)
    #print(all_transitions)

    # Crear diagrama de Sankey
    #plot_sankey(all_transitions,group_to_index)
    for num,month_trans in enumerate(transitions):
        plot_sankey(month_trans,"Sankey diagram for users' group transition from the month of: "+df.columns[num]+'__'+df.columns[num+1])

    #plot_sankey(all_transitions,"Sankey diagram for users' group transitions throughout the year")



#def plot_sankey(transitions,group_to_index):
def plot_sankey(transitions,title):

    num_nodes = 5+1

    colors = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 'rgba(44, 160, 44, 0.8)',
              'rgba(214, 39, 40, 0.8)', 'rgba(148, 103, 189, 0.8)']
    node_colors = [colors[i % len(colors)] for i in range(num_nodes)]
    link_colors = [node_colors[source] for source in transitions['source']]
        
    fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        #label=[f"Group {i}" for i in sorted(group_to_index.keys())],
        label=[f"Group {i}" for i in [x for x in range(num_nodes)]],
        color=node_colors,
    ),
    link=dict(
        arrowlen=15,
        source=transitions.iloc[:,0],
        target=transitions.iloc[:,1],
        value=transitions.iloc[:,2],
        color=link_colors
    )
    )])

    fig.update_layout(title_text=title, font_size=10)
    fig.show()
    fig.write_html(BASE_DIR+f'/Experiment_2/results/graphs/\sankey/sankey_{title[-8:]}.html')
'''






def sankey_diagram(df):
    # Generar transiciones
    transitions = []
    for i in range(len(df.columns)-1):
        start_month = df.columns[i]
        end_month = df.columns[i + 1]
        transition = df.groupby([start_month, end_month]).size().reset_index(name='count')
        transition.columns = ['source', 'target', 'value']
        transitions.append(transition)
    
    # Crear mapeo de grupos a índices
    all_groups = sorted(set(df.values.flatten()))
    group_to_index = {group: idx for idx, group in enumerate(all_groups)}

    # Ajustar transiciones con índices
    for transition in transitions:
        transition['source'] = transition['source'].map(group_to_index)
        transition['target'] = transition['target'].map(group_to_index)
    
    # Crear diagrama de Sankey
    for num, month_trans in enumerate(transitions):
        start_month = df.columns[num]
        end_month = df.columns[num + 1]
        plot_sankey(
            month_trans,
            group_to_index,
            f"Sankey diagram for users' group transition from the month of: {start_month} to {end_month}"
        )

def plot_sankey(transitions, group_to_index, title):
    num_nodes = len(group_to_index)

    colors = [
        'rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 'rgba(44, 160, 44, 0.8)',
        'rgba(214, 39, 40, 0.8)', 'rgba(148, 103, 189, 0.8)', 'rgba(140, 86, 75, 0.8)',
        'rgba(227, 119, 194, 0.8)', 'rgba(127, 127, 127, 0.8)', 'rgba(188, 189, 34, 0.8)',
        'rgba(23, 190, 207, 0.8)'
    ]
    node_colors = [colors[i % len(colors)] for i in range(num_nodes)]
    link_colors = [node_colors[source] for source in transitions['source']]
        
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[f"Group {i}" for i in group_to_index.keys()],
            color=node_colors,
        ),
        link=dict(
            arrowlen=15,
            source=transitions['source'],
            target=transitions['target'],
            value=transitions['value'],
            color=link_colors
        )
    )])

    fig.update_layout(title_text=title, font_size=10)
    #fig.show()





sankey_diagram(group_per_month)

















     


