import shutil
from pathlib import Path

source_dir = Path('/home/ubuntu/attachments')
dest_dir = Path('/home/ubuntu/repos/Navigator/data')

# Map of actual filenames to required filenames
file_mapping = {
    '2022.091200.xls': '2022.091200.xls',  # Keep original name
    '20229.xlsx': '深加工二车间2022年9月份生产日报表.xlsx',
    '2023-510.xls': '总汇表：二车间生产月报表2023-5(10).xls',
    '2023201.xlsx': '2023年5月熔炼加加工二工车间产量统计报表.xlsx',
    '20235.xlsx': '2023年5月熔洗1200吨趋势生产统计报表.xlsx',
    '2023.xlsx': '深加工二车间2023年月报.xlsx',
    '20232.xlsx': '2023年规格统计(2).xlsx',
    '2023+.xlsx': '深加工制品数量2023年5月.xlsx',
    '20235120017.xlsx': '20235120017.xlsx'  # Keep original name
}

dest_dir.mkdir(exist_ok=True)

for source_name, dest_name in file_mapping.items():
    source = source_dir / source_name
    dest = dest_dir / dest_name
    if source.exists():
        print(f"Copying {source_name} -> {dest_name}")
        shutil.copy2(source, dest)
    else:
        print(f"Warning: Source file not found: {source_name}")
