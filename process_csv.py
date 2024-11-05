import os
import pandas as pd

if __name__ == '__main__':
    filePath_runoff = 'data/CAMELS-US/runoff'
    save_path = 'data/camels.csv'
    csv_file_runoff = os.listdir(filePath_runoff)
    csv_file_runoff.sort()
    selected_files = csv_file_runoff[0:]
    merged_data = pd.DataFrame()
    df = pd.read_csv(filePath_runoff + '/' + "01013500.csv")
    column = df.iloc[:, 0]
    merged_data = pd.concat([merged_data, column], axis=1)
    index = 0
    for runoff in selected_files:
        print(runoff)
        df = pd.read_csv(filePath_runoff + '/' + runoff)
        file_name = runoff.split('.')[0]
        df.rename(columns={'runoff': file_name}, inplace=True)

        column = df.iloc[:, 1]
        merged_data = pd.concat([merged_data, column], axis=1)
        # 保存合并后的文件
        index += 1
    merged_data.to_csv(save_path, encoding='utf-8', header=True, index=False)
