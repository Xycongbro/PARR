import pandas as pd
import os

# 设置起始和结束日期
start_date = '1989-10-01'
end_date = '2008-09-30'

# 设置文件夹路径
folder_path = '/data2/zx/dataset/CAMELS-US/runoff'

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, filename+'_new.csv')
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 确保日期列是日期格式
        df['date'] = pd.to_datetime(df['date'])

        # 截取指定日期范围内的数据
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        filtered_df = df.loc[mask]

        # 保存截取后的数据为新的CSV文件
        filtered_df.to_csv(file_path, index=False)
        print(f'Processed and saved: {file_path}')