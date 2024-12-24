# 嫌疑人轨迹预测系统

## 项目简介
这是一个基于深度学习的嫌疑人轨迹预测系统，使用时空多关系图卷积网络（STMRGCN）进行轨迹预测。该系统能够通过分析历史轨迹数据，预测嫌疑人未来可能的移动路径，为执法部门提供决策支持。

## 主要特点
- 基于深度学习的轨迹预测
- 多关系图卷积网络支持
- 控制点预测功能
- 轨迹优化算法
- 高质量可视化输出
- 支持中文界面

## 安装教程
1. 克隆仓库：
```bash
git clone [仓库地址]
cd suspect_tracking
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据：
将轨迹数据文件放置在 `datasets` 目录下的相应文件夹中。

## 文件结构
```
suspect_tracking/
├── models/                 # 模型实现
│   ├── stmrgcn.py         # 时空多关系图卷积网络模型
│   ├── graph_conv.py      # 图卷积层实现
│   └── graph_tern.py      # 图注意力机制
├── utils/                  # 工具函数
│   ├── dataloader.py      # 数据加载器
│   ├── visualizer.py      # 可视化工具
│   ├── loss.py           # 损失函数
│   └── augmentor.py      # 数据增强
├── config/                # 配置文件
│   ├── default.yaml      # 默认配置
│   └── train_config.yaml # 训练配置
├── train.py              # 训练脚本
├── test.py               # 测试脚本
└── evaluate.py           # 评估脚本
```

## 使用方法

### 训练模型
```bash
python train.py --config config/train_config.yaml
```

### 测试模型
```bash
python test.py --tag [模型标签] --n_samples [样本数量]
```

### 评估模型
```bash
python evaluate.py --model_path [模型路径]
```

## 可视化示例
系统提供两种可视化输出：
1. 轨迹预测可视化：显示观测轨迹、真实轨迹和预测轨迹
2. 控制点可视化：显示预测的关键控制点和路径

可视化结果保存在 `checkpoints/[实验标签]/` 目录下。

## 配置说明
在 `config` 目录下的 YAML 文件中可以调整以下参数：
- 模型参数：图卷积层数、隐藏层维度等
- 训练参数：学习率、批次大小、训练轮数等
- 数据参数：序列长度、采样间隔等
- 可视化参数：图表样式、保存格式等

## 注意事项
1. 确保安装了所有必要的依赖包
2. 数据格式需符合系统要求
3. 建议使用GPU进行训练
4. 可视化输出支持中文显示，无需额外配置

## 性能指标
系统使用以下指标评估预测性能：
- ADE (Average Displacement Error)：平均位移误差
- FDE (Final Displacement Error)：最终位移误差

## 维护说明
- 定期更新依赖包版本
- 检查数据集完整性
- 备份训练好的模型
- 及时记录实验结果
