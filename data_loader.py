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
            # 尝试从多个可能的文件中读取温度数据
            temp_file_patterns = [
                "深加工二车间2022年9月份生产日报表.xlsx",  # Original file
                "*深加工*二车间*2022*9*.xlsx",           # Flexible pattern for original
                "*2022*9*.xls*",                      # More general pattern
                "2022.091200.xls",                    # Alternative file
                "*2022*原片1200*.xls",                 # Alternative pattern
                "*2022*.xls*"                         # Most general pattern
            ]
            temp_file = None
            for pattern in temp_file_patterns:
                try:
                    matches = list(self.data_path.glob(pattern))
                    if matches:
                        temp_file = matches[0]
                        print(f"Found temperature data file: {temp_file}")
                        break
                except Exception as e:
                    print(f"Error with pattern {pattern}: {str(e)}")
                    continue
            
            if not temp_file:
                print("Warning: Could not find temperature data file")
                return empty_data
                
            # 尝试不同的sheet名称
            possible_sheets = [
                "温度记录",          # Original sheet name
                "温度数据",          # Alternative name
                "Sheet3",          # Default sheet
                "Sheet1",          # Common default
                "Sheet2"           # Another common default
            ]
            
            temp_data = None
            for sheet_name in possible_sheets:
                try:
                    # 尝试不同的skiprows值
                    for skip_rows in [0, 1, 2]:
                        try:
                            df = pd.read_excel(temp_file, sheet_name=sheet_name, skiprows=skip_rows)
                            if not df.empty:
                                print(f"Successfully loaded sheet '{sheet_name}' with skiprows={skip_rows}")
                                temp_data = df
                                break
                        except Exception as e:
                            continue
                    if temp_data is not None:
                        break
                except Exception as e:
                    print(f"Error reading sheet {sheet_name}: {str(e)}")
                    continue
                    
            if temp_data is None:
                print(f"Warning: Could not read any sheets from {temp_file}")
                return empty_data
            
            # 如果成功读取，处理数据
            if not temp_data.empty:
                # 第一列包含温度测量点信息
                measurement_col = temp_data.columns[0]
                
                # 识别温度测量行的索引
                temp_keywords = {
                    'vault_temperature': ['拱顶', '顶部', 'vault', 'top'],
                    'bottom_temperature': ['炉底', '底部', 'bottom'],
                    'side_temperature': ['炉壁', '侧面', 'side', 'wall'],
                    'melting_temperature': ['熔化', '熔炼', 'melt']
                }
                
                # 初始化温度行索引映射
                temp_rows = {}
                for target_col, keywords in temp_keywords.items():
                    for idx, value in enumerate(temp_data[measurement_col]):
                        if isinstance(value, str) and any(keyword in value.lower() for keyword in keywords):
                            temp_rows[target_col] = idx
                            break
                
                if not temp_rows:
                    print("Warning: Could not find temperature measurements")
                    return empty_data
                
                # 创建新的DataFrame
                processed_data = pd.DataFrame()
                
                # 日期在列名中，跳过第一列（测量点描述列）
                timestamps = [col for col in temp_data.columns[1:] if isinstance(col, pd.Timestamp)]
                if not timestamps:
                    print("Warning: No valid timestamps found in columns")
                    return empty_data
                
                # 转置数据：将时间从列名变为行
                processed_data['timestamp'] = timestamps
                
                # 添加温度列，从对应的行中获取数据
                for col_name, row_idx in temp_rows.items():
                    # 获取该行的温度数据，跳过第一列（测量点描述列）
                    temp_values = temp_data.iloc[row_idx, 1:]
                    # 只取对应时间戳的数据
                    temp_values = temp_values[timestamps]
                    processed_data[col_name] = pd.to_numeric(temp_values, errors='coerce')
                
                # 删除timestamp为NaT的行
                processed_data = processed_data.dropna(subset=['timestamp'])
                processed_data = processed_data.sort_values('timestamp')
                
                # 确保所有必需的列都存在
                for col in ['vault_temperature', 'bottom_temperature', 
                           'side_temperature', 'melting_temperature']:
                    if col not in processed_data.columns:
                        processed_data[col] = None
                
                return processed_data[['timestamp', 'vault_temperature', 'bottom_temperature', 
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
            param_file_patterns = ["*二车间*2023年周报*.xlsx", "*2023年周报*.xlsx"]
            excel_file = None
            for pattern in param_file_patterns:
                matches = list(self.data_path.glob(pattern))
                if matches:
                    excel_file = matches[0]
                    break
                    
            if not excel_file:
                print("Warning: Could not find parameter data file")
                return empty_data
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
