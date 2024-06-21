import pandas as pd
import glob

# Path
carpeta = './data/dataset_auga1/train_dataset'

# Obtain csv files list
archivos_csv = glob.glob(carpeta + '/*.csv')
archivos_csv = [archivo for archivo in archivos_csv if (not archivo.endswith('train_ord.csv') and not archivo.endswith('train_ord_trans.csv'))]
#print(archivos_csv)

df = pd.DataFrame()
# Leer cada archivo CSV y concatenarlo al DataFrame
for archivo in archivos_csv:
    read = pd.read_csv(archivo)
    df = pd.concat([df, read], ignore_index=True)
#print(df)

df = df.groupby('date',sort=False)['meter_value'].apply(list).reset_index(name='consumos')
#print(df)

# Count number of users grouped
n_users = len(df['consumos'].iloc[0])
#print(n_users)

# List of total users to use as row key
list_users = []
for i in range(1,n_users+1,1):
    list_users.append(f"User_{i}")
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
print(df)
print(df.isna().sum(axis=1))

# Export results to format files easy to read and manipulate
#df.to_csv(f"./data/dataset_auga1/train_dataset/train_ord.csv", index=True, header=True)
#df.to_excel(f"./data/dataset_auga1/train_dataset/train_ord.xlsx", index=True, header=True)

#df.transpose().to_csv(f"./data/dataset_auga1/train_dataset/train_ord_trans.csv", index=True, header=True)
#df.transpose().to_excel(f"./data/dataset_auga1/train_dataset/train_ord_trans.xlsx", index=True, header=True)
