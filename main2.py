import os
files = sorted(os.listdir("/data2/zx/dataset/CAMELS-GB/forcing"), key=lambda x: int(x.split('.')[0]))  # 根据实际情况调整 split
save_path = "/data2/zx/dataset/CAMELS-GB/669_basins_list.txt"
# 创建并打开一个文本文件以写入
with open(save_path, mode='w') as file:
    for filename in files:
        a = filename.split('.')[0]
        if a == "18011" or a == "26006":
            continue
        file.write(a + '\n')  # 将每个文件名的数字部分写入文件并换行
