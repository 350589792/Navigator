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
        
        # 定义可能的sheet名称
        possible_sheets = ['生产统计', '统计', 'Sheet1', '9-9.30']
        
        try:
            # 加载2022年9月数据
            file_2022 = self.data_path / "2022.091200.xls"
            for sheet_name in possible_sheets:
                try:
                    sept_2022_data = pd.read_excel(file_2022, sheet_name=sheet_name)
                    production_data['2022_09'] = self._process_production_sheet(sept_2022_data)
                    break
                except Exception as e:
                    print(f"Warning: Could not load sheet {sheet_name} from {file_2022}: {str(e)}")
                    continue
            
            # 加载2023年5月数据
            file_2023 = self.data_path / "2023年5月熔洗1200吨趋势生产统计报表.xlsx"
            for sheet_name in possible_sheets:
                try:
                    may_2023_data = pd.read_excel(file_2023, sheet_name=sheet_name)
                    production_data['2023_05'] = self._process_production_sheet(may_2023_data)
                    break
                except Exception as e:
                    print(f"Warning: Could not load sheet {sheet_name} from {file_2023}: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"Warning: Error loading production data: {str(e)}")
        
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
                self.data_path / "2022.091200.xls",
                sheet_name="Sheet3"
            )
            
            # 如果成功读取，处理数据
            if not temp_data.empty:
                # 第一列是日期/批号，其他列是温度数据
                date_col = temp_data.columns[0]
                temp_cols = temp_data.columns[1:5]  # 取前4列温度数据
                
                # 重命名温度列
                temp_data = temp_data.rename(columns={
                    temp_cols[0]: 'vault_temperature',
                    temp_cols[1]: 'bottom_temperature',
                    temp_cols[2]: 'side_temperature',
                    temp_cols[3]: 'melting_temperature'
                })
                
                # 使用第一行的日期作为列名中的时间戳
                temp_data['timestamp'] = temp_data[date_col].fillna(method='ffill')
                temp_data = temp_data.sort_values('timestamp')
                
                return temp_data[['timestamp', 'vault_temperature', 'bottom_temperature', 
                                'side_temperature', 'melting_temperature']]
        
        except Exception as e:
            print(f"Warning: Could not load temperature data: {str(e)}")
            print("Proceeding with empty temperature dataset")
        
        return empty_data

    def load_parameter_data(self):
        """加载工艺参数数据"""
        # 创建一个空的DataFrame作为默认返回值
        empty_data = pd.DataFrame(columns=[
            'timestamp',
            'heavy_oil_flow',    # 重油流量
            'natural_gas_flow',  # 天然气流量
            'air_ratio',         # 空气比
            'air_oil_ratio',     # 空油比
            'oxygen_content'     # 氧含量
        ])
        
        try:
            # 尝试从不同的sheet名称加载数据
            excel_file = self.data_path / "二车间2023年周报（含20周）(1).xlsx"
            xls = pd.ExcelFile(excel_file)
            
            # 尝试可能的sheet名称
            possible_sheets = ['工艺参数', '参数', 'Sheet1', '第1周']
            
            for sheet_name in possible_sheets:
                try:
                    if sheet_name in xls.sheet_names:
                        param_data = pd.read_excel(excel_file, sheet_name=sheet_name)
                        
                        # 查找时间列
                        time_cols = [col for col in param_data.columns 
                                   if any(keyword in str(col) 
                                        for keyword in ['时间', '日期', 'time', 'date'])]
                        
                        if time_cols:
                            time_col = time_cols[0]
                            # 处理工艺参数
                            processed_params = pd.DataFrame()
                            processed_params['timestamp'] = pd.to_datetime(
                                param_data[time_col], errors='coerce'
                            )
                            
                            # 尝试找到对应的参数列
                            param_mapping = {
                                'heavy_oil_flow': ['重油流量', '重油', 'oil'],
                                'natural_gas_flow': ['天然气流量', '天然气', 'gas'],
                                'air_ratio': ['空气比', '空气', 'air'],
                                'air_oil_ratio': ['空油比', '油气比'],
                                'oxygen_content': ['氧含量', '氧气', 'oxygen']
                            }
                            
                            for target_col, possible_names in param_mapping.items():
                                for name in possible_names:
                                    matching_cols = [col for col in param_data.columns 
                                                   if name in str(col).lower()]
                                    if matching_cols:
                                        processed_params[target_col] = param_data[matching_cols[0]]
                                        break
                                if target_col not in processed_params.columns:
                                    processed_params[target_col] = None
                            
                            # 如果至少有一个参数列被找到，返回数据
                            if any(col in processed_params.columns for col in param_mapping.keys()):
                                return processed_params.sort_values('timestamp')
                
                except Exception as e:
                    print(f"Warning: Could not load data from sheet {sheet_name}: {str(e)}")
                    continue
            
            print("Warning: Could not find valid parameter data in any sheet")
            return empty_data
            
        except Exception as e:
            print(f"Warning: Could not load parameter data: {str(e)}")
            print("Proceeding with empty parameter dataset")
            return empty_data

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
        
        # 定义可能的列名映射
        column_mappings = {
            'date': ['日期', '时间', 'date', 'time'],
            'production_volume': ['产量', '生产量', 'production'],
            'yield_rate': ['良率', '产率', 'yield'],
            'energy_consumption': ['能耗', '能量消耗', 'energy']
        }
        
        try:
            # 对每个目标列，尝试找到匹配的源列
            for target_col, possible_names in column_mappings.items():
                found = False
                for name in possible_names:
                    matching_cols = [col for col in df.columns if name in str(col).lower()]
                    if matching_cols:
                        if target_col == 'date':
                            processed_df[target_col] = pd.to_datetime(df[matching_cols[0]], errors='coerce')
                        else:
                            processed_df[target_col] = df[matching_cols[0]]
                        found = True
                        break
                if not found:
                    processed_df[target_col] = None
            
            return processed_df
            
        except Exception as e:
            print(f"Warning: Error processing production sheet: {str(e)}")
            # 返回空的DataFrame，保持一致的列结构
            return pd.DataFrame(columns=['date', 'production_volume', 'yield_rate', 'energy_consumption'])

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
