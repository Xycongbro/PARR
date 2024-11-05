# 测试数据
# import glob
# import os.path
# import pandas as pd
#
# directory = 'data/CAMELS-US/forcing'
# files = sorted(os.listdir(directory))
# selected_files = files[1:2]
#
# # output_csv = 'data/camels/'
# # if os.path.exists(output_csv):
# #     os.remove(output_csv)
#
#
# with_header = True
# for file in selected_files:
#     df1 = pd.read_csv('data/CAMELS-US/forcing/' + file)
#     df2 = pd.read_csv('data/CAMELS-US/runoff/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     merged_df.insert(0, 'location', new_column_value)
#     # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#     merged_df.to_csv(output_csv+file, mode='a', index=False, header=with_header)
#     # merged_df.to_csv(output_csv+file, index=False, header=with_header)
#     with_header = False


# # 训练数据 camels_448_train
# import os.path
# import sys
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
# output_csv = 'data/camels_448_train.csv'
# if os.path.exists(output_csv):
#     os.remove(output_csv)
#
# df3 = pd.read_csv('/data2/zx/dataset/static/static_attributes.csv')
# with_header = True
# i = 0
# for file in open("data/448basins_list.txt"):
#     i += 1
#     file = str(file.strip()) + '.csv'
#     print(file)
#     df1 = pd.read_csv('/data2/zx/Pyraformer/data/CAMELS-US/forcing/' + file)
#     # df4 = pd.read_csv('/data2/zx/dataset/streamflow_us/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/runoff/' + file)
#     # df2 = pd.merge(df4, df2, on="date", how="inner")
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                  columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         with_header = False
#     if i >= 10:
#         break

# 读取已保存的CSV文件
# merged_df = pd.read_csv(output_csv)
# if os.path.exists(output_csv):
#     os.remove(output_csv)
# # Standardize columns 3 to 10
# scaler = StandardScaler()
# merged_df.iloc[:, 2:10] = scaler.fit_transform(merged_df.iloc[:, 2:10])
#
#
# # 将标准化后的数据保存为CSV文件
# merged_df.to_csv(output_csv, index=False)


# 448个站点信息的训练数据
# import os.path
# import sys
#
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
# output_csv = 'data/camels_448_sta_9.csv'
# if os.path.exists(output_csv):
#     os.remove(output_csv)
#
# df3 = pd.read_csv('/data2/zx/dataset/CAMELS-US/static/static_attributes.csv')
# with_header = True
# i = 0
# for file in open("data/448basins_list.txt"):
#     file = str(file.strip()) + '.csv'
#     df1 = pd.read_csv('/data2/zx/dataset/CAMELS-US/forcing/maurer/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/CAMELS-US/runoff/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         i += 1
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         # merged_df = merged_df.drop(merged_df.columns[-1], axis=1)
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                    columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         # merged_df.fillna(0, inplace=True)
#         # merged_df.dropna()
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         with_header = False
# df = pd.read_csv(output_csv)
# print(df.shape[0])
# print(i)


# # # 读取已保存的CSV文件
# merged_df = pd.read_csv(output_csv)
# print(merged_df.columns)

# if os.path.exists(output_csv):
#     os.remove(output_csv)
# # Standardize columns 3 to 10
# scaler = StandardScaler()
# merged_df.iloc[:, 2:10] = scaler.fit_transform(merged_df.iloc[:, 2:10])
#
#
# # 将标准化后的数据保存为CSV文件
# merged_df.to_csv(output_csv, index=False)


# 448测试数据
import os.path
import pandas as pd

output_path = 'data/448_sta_9/'

df3 = pd.read_csv('/data2/zx/dataset/CAMELS-US/static/static_attributes.csv')
with_header = True
i = 0
for file in open("data/448basins_list.txt"):
    file = str(file.strip()) + '.csv'
    output_csv = output_path + file
    if os.path.exists(output_csv):
        os.remove(output_csv)
    df1 = pd.read_csv('/data2/zx/dataset/CAMELS-US/forcing/maurer/' + file)
    df2 = pd.read_csv('/data2/zx/dataset/CAMELS-US/runoff/' + file)
    merged_df = pd.merge(df1, df2, on="date", how="inner")
    new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
    if int(new_column_value) in df3.iloc[:, 0].values:
        i += 1
        static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
        merged_df.insert(0, 'location', new_column_value)
        # merged_df = merged_df.drop(merged_df.columns[-1], axis=1)
        static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
        merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
        # 提取前两列
        first_two_columns = merged_df.iloc[:, :2]
        # 提取后面的几列（假设你要提取后三列）
        last_columns = merged_df.iloc[:, 2:]
        static_df = pd.DataFrame(len(merged_df) * [static_row_values],
                                   columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
        merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并

        # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])

        # merged_df = merged_df[new_cols]
        # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
        # merged_df.fillna(0, inplace=True)
        merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
        # with_header = False
print(i)


# 每个站点信息 没有静态属性
# import os.path
# import sys
#
# import pandas as pd
#
# output_path = 'data/673/'
#
#
# with_header = True
# i = 0
# for file in open("data/673basins_list.txt"):
#     file = str(file.strip()) + '.csv'
#     df1 = pd.read_csv('/data2/zx/dataset/dayment/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/runoff/' + file)
#     # merged_df = pd.merge(df1, df2, on="date", how="inner")
#     merged_df = pd.merge(df1, df2, on="date", how="outer")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     merged_df.fillna(0, inplace=True)
#     i += 1
#     merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#     merged_df.insert(0, 'location', new_column_value)
#     output_csv = output_path + new_column_value + '.csv'
#     if os.path.exists(output_csv):
#         os.remove(output_csv)
#     merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#     # with_header = False
#     # if i == 1:
#     #     sys.exit(0)
# print(i)



import os.path
import sys

# import pandas as pd
#
# output_csv = 'data/camels_673_train.csv'
# if os.path.exists(output_csv):
#     os.remove(output_csv)
#
# with_header = True
# i = 0
# for file in open("data/673basins_list.txt"):
#     i += 1
#     # if i < 475:
#     #     continue
#     file = str(file.strip()) + '.csv'
#     df1 = pd.read_csv('/data2/zx/dataset/dayment/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/runoff/' + file)
#     # merged_df = pd.merge(df1, df2, on="date", how="inner")
#     merged_df = pd.merge(df1, df2, on="date", how="outer")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     merged_df.fillna(0, inplace=True)
#     merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#     merged_df.insert(0, 'location', new_column_value)
#     merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#     with_header = False
#     # print(file)
#     # if i == 10:
#     #     sys.exit(0)
# print(i)

# # 673站点信息 没有静态属性
# import os.path
# import sys
#
# import pandas as pd
#
# output_csv = 'data/camels_673.csv'
# if os.path.exists(output_csv):
#     os.remove(output_csv)
#
# with_header = True
# i = 0
# for file in open("data/673basins_list.txt"):
#     file = str(file.strip()) + '.csv'
#     df1 = pd.read_csv('/data2/zx/dataset/dayment/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/runoff/' + file)
#     # merged_df = pd.merge(df1, df2, on="date", how="inner")
#     merged_df = pd.merge(df1, df2, on="date", how="outer")
#     merged_df.fillna(0, inplace=True)
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     i += 1
#     merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#     merged_df.insert(0, 'location', new_column_value)
#     merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#     with_header = False
# print(i)


###################################################################################
# camels-ch
# 训练数据 camels_296_train

# import os.path
# import sys
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
# directory = '/data2/zx/dataset/camels-ch/runoff-ch'
# files = sorted(os.listdir(directory))
# selected_files = files[0:296]
#
# output_csv = 'data/camels_296_train.csv'
# if os.path.exists(output_csv):
#     os.remove(output_csv)
#
# df3 = pd.read_csv('/data2/zx/dataset/camels-ch/static_attributes_ch_filtered.csv')
# with_header = True
# for file in selected_files:
#     df1 = pd.read_csv('/data2/zx/dataset/camels-ch/forcing-ch/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/camels-ch/runoff-ch/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     print(new_column_value)
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                  columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         with_header = False


# import os.path
# import pandas as pd
#
# output_path = 'data/296_2/'
#
# df3 = pd.read_csv('/data2/zx/dataset/camels-ch/static_attributes_ch_filtered.csv')
# with_header = True
# i = 0
# for file in open("data/numbers.csv"):
#     file = str(file.strip()) + '.csv'
#     output_csv = output_path + file
#     print(output_csv)
#     if os.path.exists(output_csv):
#         os.remove(output_csv)
#     df1 = pd.read_csv('/data2/zx/dataset/camels-ch/forcing-ch/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/camels-ch/runoff-ch/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         i += 1
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         # merged_df = merged_df.drop(merged_df.columns[-1], axis=1)
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                  columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         # with_header = False
# print(i)



###################################################################################
# camels-ch
# 训练数据 camels_897_train

# import os.path
# import sys
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
# directory = '/data2/zx/dataset/camels-br/runoff'
# files = sorted(os.listdir(directory))
# selected_files = files[0:897]
#
# output_csv = 'data/camels_897_train_2.csv'
# if os.path.exists(output_csv):
#     os.remove(output_csv)
#
# df3 = pd.read_csv('/data2/zx/dataset/camels-br/static/static_attributes.csv')
# with_header = True
# i = 0
# for file in selected_files:
#     i += 1
#     df1 = pd.read_csv('/data2/zx/dataset/camels-br/forcing/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/camels-br/runoff/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                  columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         # 删除 static_df 的倒数第三列
#         static_df.drop(static_df.columns[-3], axis=1, inplace=True)
#         merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#         merged_df.fillna(0, inplace=True)
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         with_header = False
#     if i == 20:
#         break


# import os.path
# import pandas as pd
#
# output_path = 'data/296_2/'
#
# df3 = pd.read_csv('/data2/zx/dataset/camels-ch/static_attributes_ch_filtered.csv')
# with_header = True
# i = 0
# for file in open("data/numbers.csv"):
#     file = str(file.strip()) + '.csv'
#     output_csv = output_path + file
#     print(output_csv)
#     if os.path.exists(output_csv):
#         os.remove(output_csv)
#     df1 = pd.read_csv('/data2/zx/dataset/camels-ch/forcing-ch/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/camels-ch/runoff-ch/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         i += 1
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         # merged_df = merged_df.drop(merged_df.columns[-1], axis=1)
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                  columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         # with_header = False
# print(i)




# camels-gb

# import os.path
# import pandas as pd
#
# output_csv = 'data/camels_gb.csv'
# if os.path.exists(output_csv):
#     os.remove(output_csv)
#
# df3 = pd.read_csv('/data2/zx/dataset/CAMELS-GB/static/static_attributes_gb_normalized.csv')
# with_header = True
# i = 0
# for file in open("/data2/zx/dataset/CAMELS-GB/basins_list.txt"):
#     i += 1
#     file = str(file.strip()) + '.csv'
#     df1 = pd.read_csv('/data2/zx/dataset/CAMELS-GB/forcing/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/CAMELS-GB/runoff/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                  columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         merged_df = pd.concat([first_two_columns, static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         with_header = False


# import os.path
# import pandas as pd
#
# output_path = 'data/gb/'
#
# df3 = pd.read_csv('/data2/zx/dataset/CAMELS-GB/static/static_attributes_gb_normalized.csv')
# with_header = True
# i = 0
# for file in open("/data2/zx/dataset/CAMELS-GB/basins_list.txt"):
#     file = str(file.strip()) + '.csv'
#     output_csv = output_path + file
#     if os.path.exists(output_csv):
#         os.remove(output_csv)
#     df1 = pd.read_csv('/data2/zx/dataset/CAMELS-GB/forcing/' + file)
#     df2 = pd.read_csv('/data2/zx/dataset/CAMELS-GB/runoff/' + file)
#     merged_df = pd.merge(df1, df2, on="date", how="inner")
#     new_column_value = os.path.splitext(file)[0]  # 提取文件名的前缀
#     if int(new_column_value) in df3.iloc[:, 0].values:
#         i += 1
#         static_row = df3[df3.iloc[:, 0] == int(new_column_value)]
#         merged_df.insert(0, 'location', new_column_value)
#         # merged_df = merged_df.drop(merged_df.columns[-1], axis=1)
#         static_row_values = static_row.values[0, 1:]  # 获取selected_row的第一行数据，作为应用的值
#         merged_df['date'] = pd.to_datetime(merged_df['date']).dt.strftime('%Y-%m-%d')
#         # 提取前两列
#         first_two_columns = merged_df.iloc[:, :2]
#         # 提取后面的几列（假设你要提取后三列）
#         last_columns = merged_df.iloc[:, 2:]
#         static_df = pd.DataFrame(len(merged_df) * [static_row_values],
#                                    columns=static_row.columns[1:])  # 创建一个新的DataFrame，每行的值都是selected_row的值
#         merged_df = pd.concat([first_two_columns,static_df, last_columns], axis=1)  # 将 static_df 和 merged_df 按列合并
#
#         # new_cols = list(merged_df.columns[8:10]) + list(merged_df.columns[:8]) + list(merged_df.columns[10:])
#
#         # merged_df = merged_df[new_cols]
#         # 追加到输出文件，如果是第一个文件则包含表头，否则不包含
#         merged_df.fillna(0, inplace=True)
#         merged_df.to_csv(output_csv, mode='a', index=False, header=with_header)
#         # with_header = False
# print(i)
