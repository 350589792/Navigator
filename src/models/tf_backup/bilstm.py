import os
import random
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 固定随机种子
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# 设置结果保存路径
denoise_folder = "去噪结果"
os.makedirs(denoise_folder, exist_ok=True)

# 加载加噪数据
noise_data_folder = "加噪数据"
noise_types = ["高斯噪声", "泊松噪声", "均匀噪声", "盐椒噪声"]


# 定义 BiLSTM 模型
def build_bilstm_model(input_shape):
    inputs = Input(shape=input_shape)

    # BiLSTM 层
    bilstm_output = Bidirectional(LSTM(64, return_sequences=True))(inputs)

    # 输出层
    output = Dense(1, activation="linear")(bilstm_output)

    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


# 训练和去噪
for noise_type in noise_types:
    print(f"正在处理噪声类型: {noise_type}")

    # 加载数据
    data_path = os.path.join(noise_data_folder, f"{noise_type}_加噪数据.npz")
    data = np.load(data_path)
    original_data = data["original_data"][::100]
    noisy_data = data["noisy_data"][::100]

    # 数据归一化
    original_data = original_data[..., np.newaxis] / np.max(np.abs(original_data))
    noisy_data = noisy_data[..., np.newaxis] / np.max(np.abs(noisy_data))

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(noisy_data, original_data, test_size=0.2, random_state=42)

    # 构建模型
    model = build_bilstm_model(input_shape=x_train.shape[1:])
    model_name = f"BiLSTM_{noise_type}"
    model_folder = os.path.join(denoise_folder, model_name)
    os.makedirs(model_folder, exist_ok=True)

    # 训练模型
    model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1, validation_data=(x_test, y_test))

    # 去噪测试数据
    denoised_data = model.predict(x_test)

    # 计算评价指标
    mse = mean_squared_error(y_test.flatten(), denoised_data.flatten())
    mae = mean_absolute_error(y_test.flatten(), denoised_data.flatten())
    r2 = r2_score(y_test.flatten(), denoised_data.flatten())

    # 保存评价结果
    result_file = os.path.join(model_folder, f"{model_name}_去噪评价结果.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"噪声类型: {noise_type}\n")
        f.write(f"均方误差 (MSE): {mse:.4f}\n")
        f.write(f"平均绝对误差 (MAE): {mae:.4f}\n")
        f.write(f"判定系数 (R²): {r2:.4f}\n")

    # ============================
    # 可视化去噪效果
    # ============================
    idx = 0  # 测试集中选择一条数据
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(x_test[idx].flatten(), label="加噪数据")
    plt.title("加噪数据")
    plt.grid(True)
    plt.subplot(3, 1, 2)
    plt.plot(denoised_data[idx].flatten(), label="去噪数据", color="green")
    plt.title("去噪数据")
    plt.grid(True)
    plt.subplot(3, 1, 3)
    plt.plot(y_test[idx].flatten(), label="原始数据", color="orange")
    plt.title("原始数据")
    plt.grid(True)
    plt.suptitle(f"{noise_type} 去噪效果")
    plt.tight_layout()

    # 保存去噪效果图像
    output_image_path = os.path.join(model_folder, f"{model_name}_去噪对比.png")
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ============================
    # 可视化残差分布
    # ============================
    residuals = y_test - denoised_data
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals.flatten(), kde=True, bins=50, color='blue', alpha=0.7)
    plt.title(f"{noise_type} 残差分布")
    plt.xlabel("残差值")
    plt.ylabel("频数")
    plt.grid()
    plt.savefig(os.path.join(model_folder, f"{model_name}_残差分布.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ============================
    # 可视化误差趋势
    # ============================
    error = (y_test - denoised_data).flatten()
    plt.figure(figsize=(12, 6))
    plt.plot(error[:100], label='误差值', color='red', linestyle='--', alpha=0.8)
    plt.title(f"{noise_type} - 前100个样本误差值趋势")
    plt.xlabel("样本索引")
    plt.ylabel("误差值")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(model_folder, f"{model_name}_误差趋势.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"噪声 {noise_type} 的去噪结果已保存到: {model_folder}")

print("所有噪声的去噪任务完成！")
