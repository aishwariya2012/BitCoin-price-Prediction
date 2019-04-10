import pandas as pd
df = pd.read_csv('C:\\Users\\Aishwariya\\Desktop\\CMPE256 Project\\test.csv')
df

df.fillna(method='ffill', inplace=True)
df