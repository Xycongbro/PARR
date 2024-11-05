# import sys
#
# import pandas as pd
#
# data_frame = pd.read_csv("./data/app_zone_rpc_hour_encrypted.csv", names=['app_name', 'zone', 'time', 'value'],
#                          parse_dates=True)
# grouped_data = list(data_frame.groupby(["app_name", "zone"]))
# unique_combinations = len(grouped_data)
# print("Number of unique [app_name, zone] combinations:", unique_combinations)
# for i in range(len(grouped_data)):
#     single_df = grouped_data[i][1].drop(labels=['app_name', 'zone'], axis=1).sort_values(by="time", ascending=True)
#     # print(single_df)
#     sys.exit(0)


import pickle
import numpy as np

def compare_pkl_files(file1, file2):
    with open(file1, 'rb') as f1:
        data1 = pickle.load(f1)

    with open(file2, 'rb') as f2:
        data2 = pickle.load(f2)
    print(data1)
    print(data2)
    # 使用numpy.array_equal来比较两个数组
    return np.array_equal(data1, data2)

# 替换 'file1.pkl' 和 'file2.pkl' 为你的文件路径
file1_path = 'data/normal_us_sta_9.pkl'
file2_path = 'data/normal_us_sta_9_inverse.pkl'

if compare_pkl_files(file1_path, file2_path):
    print("两个文件内容相同。")
else:
    print("两个文件内容不同。")