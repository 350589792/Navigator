from pathlib import Path

required_files = [
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

def validate_files():
    data_path = Path("data")
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        exit(1)
    else:
        print("All required files are present!")
        
if __name__ == "__main__":
    validate_files()
