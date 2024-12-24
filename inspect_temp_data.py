import pandas as pd
from pathlib import Path

def inspect_sheet(df, sheet_name):
    """检查单个工作表的内容，专注于温度相关数据"""
    print(f"\n工作表 '{sheet_name}' 数据形状: {df.shape}")
    
    # 搜索温度相关关键词
    keywords = [
        '温度', '拱顶', '炉底', '炉壁', '熔化', 
        'temp', 'vault', 'bottom', 'side', 'melt',
        '℃', '度', '热', '温', '窑', '炉温',
        '顶温', '底温', '侧温', '熔'
    ]
    
    # 检查列名中的关键词
    temp_cols = []
    for col in df.columns:
        col_str = str(col)
        if any(keyword in col_str for keyword in keywords):
            temp_cols.append(col)
            print(f"\n发现温度相关列: {col}")
            # 显示该列的非空值统计
            non_null_count = df[col].count()
            if non_null_count > 0:
                print(f"非空值数量: {non_null_count}")
                print("前3个非空值示例:")
                print(df[col].dropna().head(3).to_string())
    
    # 在数据内容中搜索关键词
    for col in df.columns:
        col_data = df[col].astype(str)
        matches = col_data.str.contains('|'.join(keywords), case=False, na=False)
        if matches.any():
            match_count = matches.sum()
            print(f"\n列 '{col}' 中发现 {match_count} 行包含温度相关关键词")
            if match_count > 0:
                # 只显示前3个匹配的例子
                print("示例匹配:")
                matching_rows = df[matches][[col]].head(3)
                print(matching_rows.to_string())

def inspect_temperature_data(file_path):
    """详细检查温度数据文件的所有工作表，专注于温度数据"""
    try:
        print(f"\n{'='*50}")
        print(f"检查文件: {file_path}")
        print(f"{'='*50}")
        
        # 读取所有sheet
        xls = pd.ExcelFile(file_path)
        print("\n工作表列表:")
        for sheet in xls.sheet_names: 
            print(f"- {sheet}")
        
        # 检查每个sheet
        found_temp_data = False
        for sheet_name in xls.sheet_names:
            print(f"\n----- 检查工作表: {sheet_name} -----")
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                if df.empty:
                    print("工作表为空")
                    continue
                inspect_sheet(df, sheet_name)
            except Exception as e:
                print(f"读取工作表 {sheet_name} 时出错: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Error inspecting file {file_path}: {str(e)}")

def main():
    # 检查所有可能的温度数据文件
    data_files = [
        "data/2022.091200.xls",
        "data/2022.09原片1200吨项目生产统计报表.xls",
        "data/2023年5月熔洗1200吨趋势生产统计报表.xlsx",
        "data/2023年5月熔炼加加工二工车间产量统计报表.xlsx",
        "data/深加工二车间2022年9月份生产日报表.xlsx",
        "data/总汇表：二车间生产月报表2023-5(10).xls",
        "data/深加工二车间2023年月报.xlsx"
    ]
    
    
    for file_path in data_files:
        file_path = Path(file_path)
        if file_path.exists():
            inspect_temperature_data(file_path)
        else:
            print(f"\n文件不存在: {file_path}")

if __name__ == "__main__":
    main()
