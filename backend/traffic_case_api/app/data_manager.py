from typing import List, Dict
import json
import os
from pathlib import Path

class CaseDataManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cases_file = self.data_dir / "cases.json"
        
    async def get_all_cases(self) -> List[Dict]:
        """获取所有案例"""
        if not self.cases_file.exists():
            return []
            
        with open(self.cases_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    async def save_cases(self, cases: List[Dict]) -> None:
        """保存案例数据"""
        with open(self.cases_file, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
            
    async def clear_database(self) -> None:
        """清空数据库"""
        if self.cases_file.exists():
            self.cases_file.unlink()
