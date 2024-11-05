import os

import pandas as pd

# 使用示例
directory_path = "/data2/zx/dataset/CAMELS-GB/runoff"  # 替换为你的CSV文件夹路径
train_start_date_threshold = pd.Timestamp("1993-10-01")  # 设置非 NaN 日期阈值为 YYYY-MM-DD 格式
train_end_date_threshold = pd.Timestamp("2003-09-30")  # 设置非 NaN 日期阈值为 YYYY-MM-DD 格式
valid_start_date_threshold = pd.Timestamp("2003-10-01")  # 设置非 NaN 日期阈值为 YYYY-MM-DD 格式
valid_end_date_threshold = pd.Timestamp("2005-09-30")  # 设置非 NaN 日期阈值为 YYYY-MM-DD 格式
test_start_date_threshold = pd.Timestamp("2005-10-01")  # 设置非 NaN 日期阈值为 YYYY-MM-DD 格式
test_end_date_threshold = pd.Timestamp("2015-09-30")  # 设置非 NaN 日期阈值为 YYYY-MM-DD 格式
# 处理文件
files = sorted(os.listdir(directory_path), key=lambda x: int(x.split('.')[0]))  # 根据实际情况调整 split
a = 0
with open("/data2/zx/dataset/CAMELS-GB/basins_list.txt", "w") as f:
    for file_name in files:
        print(file_name)
        train_count = 0
        valid_count = 0
        test_count = 0
        df = pd.read_csv(directory_path + "/" + file_name)
        for i in range(0, len(df)):
            if pd.isna(df.iloc[i, 1]):
                continue
            current_date = pd.to_datetime(df.iloc[i, 0])
            # 检查日期范围并计数
            if train_start_date_threshold <= current_date <= train_end_date_threshold:
                train_count += 1
            elif valid_start_date_threshold <= current_date <= valid_end_date_threshold:
                valid_count += 1
            elif test_start_date_threshold <= current_date <= test_end_date_threshold:
                test_count += 1
        print(train_count, valid_count, test_count)
        if train_count >= 3200 and valid_count >= 600 and test_count >= 3200:
            if file_name == "18011.csv" or file_name == "26006.csv":
                continue
            a += 1
            print(a)
            f.write(file_name.split('.')[0] + "\n")

    # 输出当前文件的统计结果
    # print(f"File: {file_name}")
    # print(f"Train rows: {train_count}")
    # print(f"Valid rows: {valid_count}")
    # print(f"Test rows: {test_count}")
    # print("-----")
print(a)
