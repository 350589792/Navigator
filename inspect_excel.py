import pandas as pd
from pathlib import Path

def inspect_excel_file(file_path):
    """检查Excel文件的结构"""
    try:
        print(f"\n检查文件: {file_path}")
        xls = pd.ExcelFile(file_path)
        print("工作表列表:")
        for sheet in xls.sheet_names:
            print(f"  - {sheet}")
            
        # 读取第一个sheet来查看列名
        first_sheet = xls.sheet_names[0]
        df = pd.read_excel(file_path, sheet_name=first_sheet)
        print(f"\n第一个工作表 '{first_sheet}' 的列名:")
        for col in df.columns:
            print(f"  - {col}")
            
    except Exception as e:
        print(f"Error inspecting {file_path}: {str(e)}")

def main():
    data_dir = Path("data")
    excel_files = list(data_dir.glob("*.xls*"))
    
    for file in excel_files:
        inspect_excel_file(file)

if __name__ == "__main__":
    main()
