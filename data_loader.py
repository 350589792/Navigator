import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    def __init__(self, data_path):
        """
        初始化数据加载器
        Args:
            data_path: 数据文件根路径
        """
        self.data_path = Path(data_path)
        self.required_files = [
            "2022.09原片1200吨项目生产统计报表.xls",
            "2023年5月熔洗1200吨趋势生产统计报表.xlsx",
            "2023年5月熔炼加加工二工车间产量统计报表.xlsx",
            "2023年规格统计(2).xlsx",
            "二车间2023年周报（含20周）(1).xlsx",
            "深加工二车间2022年9月份生产日报表.xlsx",
            "深加工二车间2023年月报.xlsx",
            "深加工制品数量2023年5月.xlsx",
            "总汇表：二车间生产月报表2023-5(10).xls"
        ]

    def validate_files(self):
        """验证所需文件是否存在"""
        missing_files = []
        for file in self.required_files:
            if not (self.data_path / file).exists():
                missing_files.append(file)
        if missing_files:
            raise FileNotFoundError(f"缺少以下文件: {', '.join(missing_files)}")

    def load_production_data(self):
        """加载生产数据"""
        production_data = {}

        # 加载2022年9月数据
        sept_2022_data = pd.read_excel(
            self.data_path / "2022.09原片1200吨项目生产统计报表.xls",
            sheet_name="生产统计"
        )
        production_data['2022_09'] = self._process_production_sheet(sept_2022_data)

        # 加载2023年5月数据
        may_2023_data = pd.read_excel(
            self.data_path / "2023年5月熔洗1200吨趋势生产统计报表.xlsx",
            sheet_name="趋势统计"
        )
        production_data['2023_05'] = self._process_production_sheet(may_2023_data)

        return production_data

    def load_temperature_data(self):
        """加载温度相关数据"""
        # 创建一个空的DataFrame作为默认返回值
        empty_data = pd.DataFrame(columns=[
            'timestamp',
            'vault_temperature',  # 拱顶温度
            'bottom_temperature',  # 炉底温度
            'side_temperature',  # 炉壁温度
            'melting_temperature'  # 熔化温度
        ])
        
        try:
            # 尝试从Sheet3读取温度数据
            temp_data = pd.read_excel(
                self.data_path / "2022.09原片1200吨项目生产统计报表.xls",
                sheet_name="Sheet3"
            )
            
            # 如果成功读取，处理数据
            if not temp_data.empty:
                # 使用日期/批号列作为时间戳
                temp_data['timestamp'] = pd.to_datetime(temp_data['日期/批号'])
                temp_data = temp_data.sort_values('timestamp')
                
                # 使用前4个数据列作为温度数据
                columns = temp_data.columns[1:5]  # 跳过日期/批号列，取接下来的4列
                temp_data = temp_data.rename(columns={
                    columns[0]: 'vault_temperature',
                    columns[1]: 'bottom_temperature',
                    columns[2]: 'side_temperature',
                    columns[3]: 'melting_temperature'
                })
                
                return temp_data[['timestamp', 'vault_temperature', 'bottom_temperature', 
                                'side_temperature', 'melting_temperature']]
        
        except Exception as e:
            print(f"Warning: Could not load temperature data: {str(e)}")
            print("Proceeding with empty temperature dataset")
        
        return empty_data

    def load_parameter_data(self):
        """加载工艺参数数据"""
        param_data = pd.read_excel(
            self.data_path / "二车间2023年周报（含20周）(1).xlsx",
            sheet_name="工艺参数"
        )

        # 处理工艺参数
        processed_params = pd.DataFrame()
        processed_params['timestamp'] = pd.to_datetime(param_data['记录时间'])
        processed_params['heavy_oil_flow'] = param_data['重油流量']
        processed_params['natural_gas_flow'] = param_data['天然气流量']
        processed_params['air_ratio'] = param_data['空气比']
        processed_params['air_oil_ratio'] = param_data['空油比']
        processed_params['oxygen_content'] = param_data['氧含量']

        return processed_params.sort_values('timestamp')

    def merge_all_data(self):
        """合并所有相关数据"""
        # 加载各类数据
        temp_data = self.load_temperature_data()
        param_data = self.load_parameter_data()
        production_data = self.load_production_data()

        # 将所有数据基于时间戳合并
        merged_data = pd.merge_asof(
            temp_data,
            param_data,
            on='timestamp',
            direction='nearest'
        )

        # 添加时间特征
        merged_data['hour'] = merged_data['timestamp'].dt.hour
        merged_data['day'] = merged_data['timestamp'].dt.day
        merged_data['month'] = merged_data['timestamp'].dt.month
        merged_data['year'] = merged_data['timestamp'].dt.year

        return merged_data

    def _process_production_sheet(self, df):
        """处理生产统计表数据"""
        processed_df = pd.DataFrame()
        processed_df['date'] = pd.to_datetime(df['日期'])
        processed_df['production_volume'] = df['产量']
        processed_df['yield_rate'] = df['良率']
        processed_df['energy_consumption'] = df['能耗']
        return processed_df

    def get_feature_list(self):
        """获取所有特征列表"""
        return [
            # 温度特征
            'vault_temperature',  # 拱顶温度
            'bottom_temperature',  # 炉底温度
            'side_temperature',  # 炉壁温度
            'melting_temperature',  # 熔化温度

            # 工艺参数
            'heavy_oil_flow',  # 重油流量
            'natural_gas_flow',  # 天然气流量
            'air_ratio',  # 空气比
            'air_oil_ratio',  # 空油比
            'oxygen_content',  # 氧含量

            # 时间特征
            'hour',
            'day',
            'month',
            'year'
        ]

    def save_processed_data(self, data, filename):
        """保存处理后的数据"""
        output_path = self.data_path / 'processed'
        output_path.mkdir(exist_ok=True)
        data.to_csv(output_path / filename, index=False)
