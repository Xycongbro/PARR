import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM


# 生成周期性信号
# 利用正弦函数创建周期性信号。每个周期的幅度和周期时间通过参数 T 和 A 控制。
def generate_sin(x, T, A):
    """Generate a mixed sinusoidal sequence"""
    y = np.zeros(len(x))
    for i in range(len(T)):
        y += A[i] * np.sin(2 * np.pi / T[i] * x)

    return y


# 生成协变量
# 为每个时间点生成协变量，包括时间（如星期几、小时数、月份）和序列的索引。
def gen_covariates(x, index):
    """Generate covariates"""
    covariates = np.zeros((x.shape[0], 4))
    covariates[:, 0] = (x // 24) % 7
    covariates[:, 1] = x % 24
    covariates[:, 2] = (x // (24 * 30)) % 12
    covariates[:, 0] = covariates[:, 0] / 6
    covariates[:, 1] = covariates[:, 1] / 23
    covariates[:, 2] = covariates[:, 2] / 11

    covariates[:, -1] = np.zeros(x.shape[0]) + index
    return covariates


# 生成分数布朗噪声
# 使用分数布朗运动（FBM）来生成具有长期依赖性的噪声。分数布朗运动是一种统计特性随时间变化的随机过程。
def fractional_brownian_noise(length, hurst, step):
    """Genereate fractional brownian noise"""
    f = FBM(length, hurst, step)
    noise = f.fbm()
    return noise


# 合成数据
# 创建一个包含周期性信号、协变量和分数布朗噪声的合成数据集。
# 数据集包含多个序列，每个序列由周期性信号、生成的协变量和多变量正态分布噪声组合而成。
# 该数据集保存为一个NumPy数组文件。
def synthesis_data():
    """synthesis a mixed sinusoidal dataset"""
    # 定义了三个不同的周期长度 一天24小时 一周168小时 一个月720小时
    T = [24, 168, 720]
    # 生成60个独立的时间序列
    seq_num = 60
    # 每个时间序列的长度是最长周期的20倍
    seq_len = T[-1] * 20
    data = []
    covariates = []
    for i in range(seq_num):
        start = int(np.random.uniform(0, T[-1]))
        x = start + np.arange(seq_len)
        A = np.random.uniform(5, 10, 3)
        y = generate_sin(x, T, A)
        data.append(y)
        covariates.append(gen_covariates(x, i))
        # plt.plot(x[:T[-1]], y[:T[-1]])
        # plt.show()

    data = np.array(data)
    mean, cov = polynomial_decay_cov(seq_len)

    noise = multivariate_normal(mean, cov, seq_num)
    data = data + noise
    covariates = np.array(covariates)
    data = np.concatenate([data[:, :, None], covariates], axis=2)
    np.save('data/synthetic.npy', data)


# 计算数据协方差
# 计算数据的协方差矩阵。
def covariance(data):
    """compute the covariance of the data"""
    data_mean = data.mean(0)
    data = data - data_mean
    length = data.shape[1]
    data_covariance = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            data_covariance[i, j] = (data[:, i] * data[:, j]).mean()

    return data_covariance


# 测试分数布朗噪声的协方差
# 生成分数布朗噪声的样本并计算其协方差矩阵，以观察其统计特性。
def test_fbm():
    """Plot the covariance of the generated fractional brownian noise"""
    f = FBM(300, 0.3, 1)
    fbm_data = []
    for i in range(100):
        sample = f.fbm()
        fbm_data.append(sample[1:])
    fbm_data = np.array(fbm_data)
    cov = covariance(fbm_data)
    plt.imshow(cov)
    plt.savefig('fbm_cov.jpg')


# 定义多项式衰减协方差
# 创建一个协方差矩阵，其中协方差随时间距离的增加而衰减。
def polynomial_decay_cov(length):
    """Define the function of covariance decay with distance"""
    mean = np.zeros(length)

    x_axis = np.arange(length)
    distance = x_axis[:, None] - x_axis[None, :]
    distance = np.abs(distance)
    cov = 1 / (distance + 1)
    return mean, cov


# 生成多变量正态分布噪声
# 基于给定的均值和协方差矩阵生成多变量正态分布的噪声。
def multivariate_normal(mean, cov, seq_num):
    """Generate multivariate normal distribution"""
    noise = np.random.multivariate_normal(mean, cov, (seq_num,), 'raise')

    return noise


# 代码生成并保存合成数据集。
# 这种合成数据集通常用于测试和评估时间序列模型的性能，尤其是在处理具有复杂特性（如周期性和长期依赖性）的数据时。
if __name__ == '__main__':
    synthesis_data()
