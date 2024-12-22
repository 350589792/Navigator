# main.py
import logging
from pathlib import Path
from datetime import datetime

def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("开始运行玻璃窑炉温度预测系统")

try:
    import torch
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from ai_model import GlassFurnaceModel, TemperaturePredictor
    from cfd_model import CFDSimulator
    from hybrid_model import HybridModel
    from trainer import ModelTrainer
    from evaluator import ModelEvaluator
    from config import Config
    from visualization import Visualizer
except ImportError as e:
    logger.error(f"导入模块时出错: {str(e)}")
    raise

def main():
    try:
        # 2. 加载配置
        config = Config("config.yaml")
        logger.info("配置加载完成")

        # 3. 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")

        # 4. 数据加载和预处理
        data_loader = DataLoader(config.get_data_config()['data_path'])
        preprocessor = DataPreprocessor()

        # 加载数据
        logger.info("开始加载数据...")
        data = data_loader.merge_all_data()
        logger.info(f"数据加载完成，共 {len(data)} 条记录")

        # 预处理数据
        logger.info("开始数据预处理...")
        # 确保创建时间特征
        data = preprocessor.create_time_features(data)
        processed_data = preprocessor.prepare_full_dataset(
            data,
            sequence_length=config.get_data_config()['sequence_length'],
            target_col=config.get_data_config()['target_column']
        )
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = processed_data
        logger.info("数据预处理完成")

        # 5. 初始化模型
        logger.info("初始化模型...")
        input_dim = X_train.shape[-1]  # 使用预处理后的特征维度
        logger.info(f"Input dimension from data: {input_dim}")
        
        # 更新模型配置
        config.config['model']['ai']['input_size'] = input_dim
        
        # AI模型
        ai_model = GlassFurnaceModel(
            input_size=input_dim,
            hidden_size=config.config['model']['ai']['hidden_size'],
            num_layers=config.config['model']['ai']['num_layers']
        ).to(device)

        # CFD模型
        cfd_model = CFDSimulator(config.config['model']['cfd'])

        # 混合模型
        hybrid_model = HybridModel(
            ai_model=ai_model,
            cfd_model=cfd_model,
            config=config.config['model']['hybrid'],
            device=device
        )

        # 6. 训练模型
        logger.info("开始模型训练...")
        trainer = ModelTrainer(hybrid_model, config.config, device)
        training_history = trainer.train(X_train, y_train, X_val, y_val)
        logger.info("模型训练完成")

        # 7. 评估模型
        logger.info("开始模型评估...")
        evaluator = ModelEvaluator(hybrid_model, config.config)
        metrics, predictions = evaluator.evaluate_model(X_test, y_test)
        logger.info("模型评估完成")

        # 8. 可视化结果
        logger.info("生成可视化结果...")
        visualizer = Visualizer(config.get_paths()['output_path'])
        visualizer.plot_training_history(training_history)
        visualizer.plot_prediction_comparison(y_test, predictions)
        visualizer.plot_error_distribution(y_test - predictions)
        logger.info("可视化完成")

        # 9. 保存结果
        logger.info("保存训练历史...")
        trainer.save_training_history(training_history)
        logger.info("运行完成")

    except Exception as e:
        logger.error(f"运行出错: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
