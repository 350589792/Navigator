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
            file_2022_patterns = ["2022.091200.xls", "*2022*原片1200*.xls"]
            file_2022 = None
            for pattern in file_2022_patterns:
                matches = list(self.data_path.glob(pattern))
                if matches:
                    file_2022 = matches[0]
                    break
            
            if file_2022:
                for sheet_name in possible_sheets:
                    try:
                        sept_2022_data = pd.read_excel(file_2022, sheet_name=sheet_name)
                        production_data['2022_09'] = self._process_production_sheet(sept_2022_data)
                        break
                    except Exception as e:
                        print(f"Warning: Could not load sheet {sheet_name} from {file_2022}: {str(e)}")
                        continue
            
            # 加载2023年5月数据
            file_2023_patterns = ["*2023*5*熔洗1200*.xlsx", "*2023*5月*.xlsx"]
            file_2023 = None
            for pattern in file_2023_patterns:
                matches = list(self.data_path.glob(pattern))
                if matches:
                    file_2023 = matches[0]
                    break
            if file_2023:
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
        """Generate synthetic temperature data with realistic patterns for glass furnace."""
        import numpy as np
        from datetime import datetime, timedelta
        
        print("Generating synthetic temperature data based on typical glass furnace operations")
        
        # Create timestamps for a month of data (hourly measurements)
        start_date = datetime(2023, 5, 1)  # May 2023 to match production data
        dates = [start_date + timedelta(hours=i) for i in range(24 * 31)]  # 31 days
        
        def generate_temp_series(base_temp, amplitude=10, trend=0):
            """Generate temperature series with realistic variations."""
            n_points = len(dates)
            
            # Daily cycle (24-hour period)
            hourly_cycle = amplitude * np.sin(2 * np.pi * np.arange(n_points) / 24)
            
            # Weekly variation (168-hour period)
            weekly_cycle = (amplitude/2) * np.sin(2 * np.pi * np.arange(n_points) / 168)
            
            # Long-term trend
            trend_component = trend * np.linspace(0, 1, n_points)
            
            # Random fluctuations
            noise = np.random.normal(0, amplitude/4, n_points)
            
            # Combine all components
            return base_temp + hourly_cycle + weekly_cycle + trend_component + noise
        
        np.random.seed(42)  # For reproducibility
        
        # Generate temperature data with realistic relationships
        # Based on typical glass furnace temperature zones
        vault_temp = generate_temp_series(1550, amplitude=15, trend=-5)    # Highest, slight cooling trend
        bottom_temp = generate_temp_series(1450, amplitude=12, trend=-3)   # Slightly lower
        side_temp = generate_temp_series(1350, amplitude=10, trend=-2)     # Lower still
        melting_temp = generate_temp_series(1300, amplitude=8, trend=-1)   # Most controlled
        
        # Create DataFrame with generated data and enforce float32 type
        temperature_data = pd.DataFrame({
            'timestamp': dates,  # Use English column name for consistency
            'vault_temperature': np.array(vault_temp, dtype=np.float32),    # 拱顶温度
            'bottom_temperature': np.array(bottom_temp, dtype=np.float32),  # 炉底温度
            'side_temperature': np.array(side_temp, dtype=np.float32),      # 炉壁温度
            'melting_temperature': np.array(melting_temp, dtype=np.float32) # 熔化温度
        })
        
        # Add some realistic constraints
        # Ensure vault temperature is always highest
        temperature_data['vault_temperature'] = temperature_data[['vault_temperature', 'bottom_temperature', 
                                                                'side_temperature', 'melting_temperature']].max(axis=1) + 50
        # Ensure melting temperature is always lowest
        temperature_data['melting_temperature'] = temperature_data[['vault_temperature', 'bottom_temperature', 
                                                                  'side_temperature', 'melting_temperature']].min(axis=1) - 50
        
        print("\nGenerated temperature data statistics:")
        print(f"Number of records: {len(temperature_data)}")
        print("\nTemperature ranges (°C):")
        # Create a mapping for display names
        display_names = {
            'vault_temperature': '拱顶温度',
            'bottom_temperature': '炉底温度',
            'side_temperature': '炉壁温度',
            'melting_temperature': '熔化温度'
        }
        for col in temperature_data.columns[1:]:
            display_name = display_names.get(col, col)
            print(f"{display_name}: {temperature_data[col].min():.1f} - {temperature_data[col].max():.1f}")
        
        return temperature_data

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
        
        print("Using empty parameter data as placeholder")
        return empty_data

    def merge_all_data(self):
        """合并所有相关数据"""
        # 加载各类数据
        temp_data = self.load_temperature_data()
        param_data = self.load_parameter_data()
        production_data = self.load_production_data()

        # 确保时间戳列是datetime类型
        if not temp_data.empty:
            temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'])
        if not param_data.empty:
            param_data['timestamp'] = pd.to_datetime(param_data['timestamp'])
        
        # 创建基础数据框架
        if temp_data.empty and param_data.empty:
            # 如果两个数据集都是空的，返回一个带有基本列的空DataFrame
            return pd.DataFrame(columns=[
                'timestamp', 'vault_temperature', 'bottom_temperature',
                'side_temperature', 'melting_temperature', 'heavy_oil_flow',
                'natural_gas_flow', 'air_ratio', 'air_oil_ratio',
                'oxygen_content', 'hour', 'day', 'month', 'year'
            ])
        
        # 如果温度数据为空但参数数据存在
        if temp_data.empty and not param_data.empty:
            merged_data = param_data
        # 如果参数数据为空但温度数据存在
        elif param_data.empty and not temp_data.empty:
            merged_data = temp_data
        else:
            # 两个数据集都有数据时进行合并
            try:
                merged_data = pd.merge_asof(
                    temp_data.sort_values('timestamp'),
                    param_data.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest'
                )
            except Exception as e:
                print(f"Warning: Error merging data: {str(e)}")
                # 如果合并失败，使用温度数据作为基础
                merged_data = temp_data
        
        # 添加时间特征
        if not merged_data.empty and 'timestamp' in merged_data.columns:
            merged_data['hour'] = merged_data['timestamp'].dt.hour
            merged_data['day'] = merged_data['timestamp'].dt.day
            merged_data['month'] = merged_data['timestamp'].dt.month
            merged_data['year'] = merged_data['timestamp'].dt.year
        
        # Ensure all numeric columns are float32 before returning
        if not merged_data.empty:
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                merged_data[col] = merged_data[col].astype(np.float32)
        
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
