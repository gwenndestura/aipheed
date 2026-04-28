import pandas as pd
df = pd.read_parquet('data/raw/corpus_raw.parquet')
print(f'Rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Sources: {df[\"fetcher_source\"].value_counts().to_dict()}')
print(f'Year range: {df[\"published\"].min()} -> {df[\"published\"].max()}')
