import sys

import pandas as pd
import os
dir_path = '/data2/zqr/CAMELS/CAMELS-US/basin_dataset_public_v1p2/basin_mean_forcing/daymet/'
save_path = '/data2/zx/dataset/dayment/'
dir_path_contents = sorted(os.listdir(dir_path))
for dir_path_content in dir_path_contents:
    path = dir_path + dir_path_content
    path_contents = sorted(os.listdir(path))
    for path_content in path_contents:
        content = path_content.split('_')[0] + '.csv'
        # 定义文件路径
        file_path = path+'/'+path_content
        try:
            # 读取文件，跳过前三行，分隔符为制表符
            df = pd.read_csv(file_path, delimiter='\s+', skiprows=3, engine='python')
            # 合并年、月、日为日期列，并转换为 datetime 类型
            df['date'] = pd.to_datetime(df[['Year', 'Mnth', 'Day']].astype(str).agg('-'.join, axis=1))

            # 删除原年、月、日和小时列
            df.drop(['Year', 'Mnth', 'Day', 'Hr'], axis=1, inplace=True)
            df.drop(['dayl(s)', 'swe(mm)'], axis=1, inplace=True)
            # 重排列，使日期列在最前
            df = df[['date'] + [col for col in df.columns if col != 'date']]
            # 显示数据帧的前几行以确认更改
            # 保存到 CSV 文件
            output_path = save_path + content
            df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue
