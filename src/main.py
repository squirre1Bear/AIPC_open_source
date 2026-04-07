import pandas as pd

df = pd.read_parquet(r"E:\AIPC_dataset\wiff\batch_1\AIPC_wiff_data_0.parquet")
pd.set_option('display.max_columns', None)
print(df.head())
print(df.columns)
print(df.info())