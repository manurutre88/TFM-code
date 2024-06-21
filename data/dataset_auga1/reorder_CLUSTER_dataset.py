import numpy as np
import pandas as pd
import os

BASE_DIR = os.getcwd()
main_path = os.path.join(BASE_DIR,'data','dataset_auga1','cluster_dataset')

def reorder_cluster_dataset(path_origin,n_cluster):

    # Import raw dataset
    df = pd.read_csv(path_origin, delimiter=',')
    #print(df)

    # Group rows by same day date
    df = df.groupby('date',sort=False)['meter_value'].apply(list).reset_index(name='consumos')
    #print(df)

    # Count number of users grouped
    n_users = len(df['consumos'].iloc[0])
    #print(n_users)

    # List of total users to use as row key
    list_users = []
    for i in range(1,n_users+1,1):
        list_users.append(f"user{i}")
    #print(list_users)

    # Stablish date as row identifier
    df.set_index('date', inplace=True)
    df.index.name = None
    #print(df)

    # Split list with meter values in several different columns
    df[list_users] = pd.DataFrame(df.consumos.tolist(), index= df.index)
    df = df.drop(columns=['consumos'])
    #print(df)

    # Transpose so as to have users as rows and consumption as columns
    df = df.transpose()
    df['label'] = n_cluster
    print(df)

    # df.to_csv(path_objective, index=False)
    return df


if __name__ == "__main__":
    
    # Reorder and append each dataset
    clusters = []
    for i in range(14): 
        cluster = reorder_cluster_dataset(main_path+f"/cluster{i}.csv", i)
        clusters.append(cluster)

    # Concat all datasets into a single one
    train_dataset = pd.concat(clusters, axis=0)

    # Rename rows
    train_dataset.index = [f'User_{i}' for i in range(1,train_dataset.shape[0]+1)]
    print(train_dataset)

    # Export to format files
    #train_dataset.to_csv("./cluster_dataset/cluster_ord.csv", index=True, header=True)
    train_dataset.to_excel(main_path+"./cluster_ord_etiquetado.xlsx", index=True, header=True)

    