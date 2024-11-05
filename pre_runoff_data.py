import sys
import pandas as pd
import os

def read_third_line(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) >= 3:
                third_line = lines[2].strip()  # 第三行内容
                return third_line
            else:
                return "文件中没有足够的行"
    except FileNotFoundError:
        return "文件未找到"


dir_path = '/data2/zqr/CAMELS/CAMELS-US/basin_dataset_public_v1p2/usgs_streamflow/'
save_path = '/data2/zx/dataset/camels-us/runoff/'
dir_path_contents = sorted(os.listdir(dir_path))
for dir_path_content in dir_path_contents:
    path = dir_path + dir_path_content
    path_contents = sorted(os.listdir(path))
    for path_content in path_contents:
        content = path_content.split('_')[0] + '.csv'
        # 定义文件路径
        file_path = path+'/'+path_content
        try:
            # 读取数据
            df = pd.read_csv(file_path, delim_whitespace=True, header=None,
                             names=['Station', 'Year', 'Month', 'Day', 'runoff', 'Quality'])
            # 合并年、月、日为日期列，并转换为 datetime 类型
            df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

            # median_value = df['runoff'][df['runoff'] != -999.00].median()
            df['runoff'] = df['runoff'].replace(-999.00, 0)
            # 计算除了 -999.00 以外的值的平均
            # valid_mean = df.loc[df['runoff'] != -999.00, 'runoff'].mean()
            # df['runoff'] = df['runoff'].replace(-999.00, median_value)

            # 删除不需要的列
            df.drop(['Station', 'Year', 'Month', 'Day', 'Quality'], axis=1, inplace=True)
            # 重排列列，使日期列在最前
            df = df[['date', 'runoff']]
            # 输出查看结果
            # print(df)
            # 如果需要保存处理后的数据到新的 CSV 文件
            # area = read_third_line("/data2/zx/Pyraformer/data/maurer/"+path_content.split('_')[0]+"_lump_maurer_forcing_leap.txt")
            # area = float(area)
            # df['runoff'] = 28316846.592 * df['runoff'] * 86400 / (area * 10 ** 6)
            save_file_path = os.path.join(save_path, content)
            df.to_csv(save_file_path, index=False)

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue


