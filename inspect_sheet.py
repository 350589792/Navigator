import pandas as pd
from pathlib import Path

def inspect_sheet_details(file_path, sheet_name):
    """详细检查指定工作表的内容"""
    try:
        print(f"\n详细检查文件: {file_path}, 工作表: {sheet_name}")
        # 读取前10行来查看实际数据结构
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=10)
        print("\n前10行数据:")
        print(df.to_string())
        
    except Exception as e:
        print(f"Error inspecting sheet {sheet_name} in {file_path}: {str(e)}")

def main():
    # 检查温度数据文件的Sheet3
    temp_file = Path("data/2022.091200.xls")
    if temp_file.exists():
        inspect_sheet_details(temp_file, "Sheet3")
    
    # 检查生产数据文件的9-9.30sheet
    prod_file = Path("data/2022.091200.xls")
    if prod_file.exists():
        inspect_sheet_details(prod_file, "9-9.30")

if __name__ == "__main__":
    main()
