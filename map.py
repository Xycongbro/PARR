import pandas as pd

gauge_dict = pd.read_csv('./data/camels_topo.txt', sep=';')
df = pd.read_csv('./data/map.csv')

# 使用 merge 函数合并 DataFrame
matching_rows = pd.merge(df, gauge_dict, left_on='basin', right_on='gauge_id')
matching_rows = matching_rows[['gauge_lat', 'gauge_lon', 'gauge_id']]
matching_rows.to_csv('./data/map.csv', index=False)