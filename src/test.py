import pandas as pd

df = pd.read_parquet('./data_guanfang/mzml_parquet/part.00000.parquet')
print(df.columns.tolist())