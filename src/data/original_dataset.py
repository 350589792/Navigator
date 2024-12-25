import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams

# 设置中文字体（以 SimHei 为例，需确保系统中有该字体）
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 文件路径
file_path = "data/SEG_45Shot_shots10-18.sgy"

# 创建保存图像、结果和加噪文件的文件夹
output_folder = "图像"
result_folder = "结果"
noise_data_folder = "加噪数据"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
os.makedirs(noise_data_folder, exist_ok=True)

# 打开 SEG-Y 文件并加载数据
with segyio.open(file_path, "r", strict=False) as sgy_file:
    traces = [trace for trace in sgy_file.trace[:]]  # 获取所有地震道
    trace_data = np.array(traces)  # 转为 NumPy 数组


# 定义添加噪声的函数
def add_gaussian_noise(data, noise_ratio=0.1):
    std_dev = noise_ratio * np.std(data)  # 根据信号标准差计算噪声强度
    noise = np.random.normal(0, std_dev, data.shape)
    return data + noise


def add_poisson_noise(data):
    noise = np.random.poisson(data - data.min())  # 确保泊松分布的正值
    return data + noise


def add_uniform_noise(data, noise_ratio=0.1):
    range_val = noise_ratio * (np.max(data) - np.min(data))  # 根据信号幅值范围计算噪声强度
    noise = np.random.uniform(-range_val, range_val, data.shape)
    return data + noise


def add_salt_and_pepper_noise(data, prob=0.01):
    noisy_data = data.copy()
    total_pixels = data.size
    num_salt = int(prob * total_pixels / 2)
    num_pepper = int(prob * total_pixels / 2)

    # 添加盐噪声（设置为最大值）
    salt_coords = np.random.choice(total_pixels, num_salt, replace=False)
    noisy_data.ravel()[salt_coords] = np.max(data)

    # 添加椒噪声（设置为最小值）
    pepper_coords = np.random.choice(total_pixels, num_pepper, replace=False)
    noisy_data.ravel()[pepper_coords] = np.min(data)

    return noisy_data


# 噪声类型
noise_functions = {
    "高斯噪声": lambda data: add_gaussian_noise(data, noise_ratio=0.1),
    "泊松噪声": add_poisson_noise,
    "均匀噪声": lambda data: add_uniform_noise(data, noise_ratio=0.1),
    "盐椒噪声": lambda data: add_salt_and_pepper_noise(data, prob=0.01),
}

# 初始化结果记录
results = []

# 保存加噪后的数据
for noise_type, noise_func in noise_functions.items():
    print(f'正在添加{noise_type}噪声！！！')
    noisy_traces = []  # 存储加噪后的所有地震道
    total_mse, total_mae, total_r2 = 0, 0, 0

    for trace in trace_data:
        noisy_trace = noise_func(trace)
        noisy_traces.append(noisy_trace)

        # 计算误差
        mse = mean_squared_error(trace, noisy_trace)
        mae = mean_absolute_error(trace, noisy_trace)
        r2 = r2_score(trace, noisy_trace)

        total_mse += mse
        total_mae += mae
        total_r2 += r2

    # 保存加噪后的数据到字典
    noisy_traces = np.array(noisy_traces)
    np.savez_compressed(
        os.path.join(noise_data_folder, f"{noise_type}_加噪数据.npz"),
        original_data=trace_data,
        noisy_data=noisy_traces,
    )

    # 计算总体平均指标
    avg_mse = total_mse / len(trace_data)
    avg_mae = total_mae / len(trace_data)
    avg_r2 = total_r2 / len(trace_data)

    # 记录结果
    results.append(
        f"总体加入 {noise_type}:\n"
        f"平均均方误差 (MSE): {avg_mse:.4f}\n"
        f"平均绝对误差 (MAE): {avg_mae:.4f}\n"
        f"平均判定系数 (R²): {avg_r2:.4f}\n"
        "-------------------------------\n"
    )

    # 仅对第一个地震道进行可视化
    if trace_data[0] is not None:
        original_trace = trace_data[0]
        noisy_trace = noise_func(original_trace)

        # 绘制对比图
        plt.figure(figsize=(10, 5))
        plt.plot(original_trace, label='原始地震道', linestyle='-', alpha=0.7)
        plt.plot(noisy_trace, label=f'加噪信号 ({noise_type})', linestyle='--', alpha=0.7)
        plt.xlabel("采样索引")
        plt.ylabel("振幅")
        plt.title(f"地震道与 {noise_type} 对比")
        plt.legend()
        plt.grid()
        plt.savefig(f'{output_folder}/地震道_{noise_type}_对比.png')
        plt.show()

# 保存结果到文件
result_file = os.path.join(result_folder, "总体加噪误差计算结果.txt")
with open(result_file, "w", encoding="utf-8") as file:
    file.writelines(results)

print(f"结果已保存到 {result_file}")
print(f"加噪后的数据已保存到 {noise_data_folder} 文件夹中。")
