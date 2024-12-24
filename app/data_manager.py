from typing import List, Dict, Any
import json
import os
from pathlib import Path

class CaseDataManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cases_file = self.data_dir / "cases.json"
        
    def transform_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """转换案例格式，统一字段名称"""
        return {
            "id": case["id"],
            "content": case["content"],
            "laws": case.get("laws") or case.get("applicable_laws", [])
        }
        
    async def get_all_cases(self) -> List[Dict]:
        """获取所有案例"""
        return self.get_all_cases_sync()
            
    def get_all_cases_sync(self) -> List[Dict]:
        """同步方式获取所有案例"""
        if not self.cases_file.exists():
            return []
            
        with open(self.cases_file, 'r', encoding='utf-8') as f:
            cases = json.load(f)
            return [self.transform_case(case) for case in cases]
            
    async def save_cases(self, cases: List[Dict]) -> None:
        """保存案例数据"""
        transformed_cases = [self.transform_case(case) for case in cases]
        with open(self.cases_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_cases, f, ensure_ascii=False, indent=2)
            
    async def clear_database(self) -> None:
        """清空数据库"""
        if self.cases_file.exists():
            self.cases_file.unlink()
