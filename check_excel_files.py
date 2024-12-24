import pandas as pd
from pathlib import Path
import sys

def check_excel_files():
    data_dir = Path('data')
    excel_files = list(data_dir.glob('*.xls*'))
    
    print("Checking Excel files for temperature data...")
    for file_path in excel_files:
        print(f"\n=== {file_path} ===")
        try:
            xls = pd.ExcelFile(file_path)
            print("Sheets:", xls.sheet_names)
            
            # Try to read each sheet and look for temperature-related columns
            for sheet in xls.sheet_names:
                print(f"\nReading sheet: {sheet}")
                df = pd.read_excel(file_path, sheet_name=sheet)
                # Print first few column names to help identify temperature data
                print("Columns:", df.columns.tolist()[:10])
                
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

if __name__ == '__main__':
    check_excel_files()
