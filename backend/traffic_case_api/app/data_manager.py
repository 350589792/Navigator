import json
import os
import aiofiles
from typing import List, Dict, Optional
from pathlib import Path

class CaseDataManager:
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.cases_file = self.data_dir / "traffic_cases.json"
        self.laws_file = self.data_dir / "traffic_laws.json"
        self.cases = []
        self.laws = {}

    @classmethod
    async def create(cls):
        """Factory method to create and initialize a CaseDataManager instance"""
        instance = cls()
        await instance._initialize_data_directory()
        instance.cases = await instance._load_cases()
        instance.laws = await instance._load_laws()
        return instance

    async def _initialize_data_directory(self):
        """Create data directory if it doesn't exist"""
        try:
            self.data_dir.mkdir(exist_ok=True)
        except PermissionError as e:
            raise OSError("无法创建数据目录：权限被拒绝。Windows系统可能需要管理员权限。") from e
        except OSError as e:
            if "WinError 123" in str(e):
                raise OSError("数据目录路径无效。请检查路径是否包含非法字符。") from e
            raise OSError(f"创建数据目录时发生错误：{str(e)}") from e
        
        # Initialize with comprehensive traffic laws if file doesn't exist
        if not self.laws_file.exists():
            basic_laws = {
                "traffic_safety_law": [
                    {
                        "law_name": "中华人民共和国道路交通安全法",
                        "article_number": "第一百一十九条",
                        "content": "违反道路交通安全法律、法规的规定，发生重大交通事故，构成犯罪的，依法追究刑事责任，并由公安机关交通管理部门吊销机动车驾驶证。"
                    },
                    {
                        "law_name": "中华人民共和国刑法",
                        "article_number": "第一百三十三条",
                        "content": "违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役。"
                    },
                    {
                        "law_name": "中华人民共和国道路交通安全法",
                        "article_number": "第七十六条",
                        "content": "机动车发生交通事故造成人身伤亡、财产损失的，由保险公司在机动车第三者责任强制保险责任限额范围内予以赔偿。"
                    }
                ],
                "criminal_law": [
                    {
                        "law_name": "中华人民共和国刑法",
                        "article_number": "第一百三十三条之一",
                        "content": "在道路上驾驶机动车，有下列情形之一的，处拘役，并处罚金：（一）追逐竞驶，情节恶劣的；（二）醉酒驾驶机动车的；（三）从事校车业务或者旅客运输，严重超过额定乘员载客，或者严重超过规定时速行驶的；（四）违反危险化学品安全管理规定运输危险化学品，危及公共安全的。"
                    }
                ]
            }
            await self._save_json(self.laws_file, basic_laws)
        
        # Initialize empty cases file if it doesn't exist
        if not self.cases_file.exists():
            await self._save_json(self.cases_file, {"cases": []})

    async def _load_cases(self) -> List[Dict]:
        """Load cases from JSON file"""
        if self.cases_file.exists():
            data = await self._load_json(self.cases_file)
            return data.get("cases", [])
        return []

    async def _load_laws(self) -> Dict:
        """Load traffic laws from JSON file"""
        if self.laws_file.exists():
            return await self._load_json(self.laws_file)
        return {}

    async def _save_json(self, file_path: Path, data: Dict):
        """Save data to JSON file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        except PermissionError as e:
            raise OSError(f"无法写入文件 {file_path}：权限被拒绝。Windows系统可能需要管理员权限。") from e
        except OSError as e:
            if "WinError 32" in str(e):
                raise OSError(f"文件 {file_path} 被另一个进程占用。请确保没有其他程序正在访问该文件。") from e
            elif "WinError 123" in str(e):
                raise OSError(f"文件路径 {file_path} 无效。请检查文件路径是否包含非法字符。") from e
            raise OSError(f"写入文件 {file_path} 时发生错误：{str(e)}") from e

    async def _load_json(self, file_path: Path) -> Dict:
        """Load data from JSON file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except PermissionError as e:
            raise OSError(f"无法读取文件 {file_path}：权限被拒绝。Windows系统可能需要管理员权限。") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"文件 {file_path} 包含无效的JSON数据：{str(e)}") from e
        except OSError as e:
            if "WinError 32" in str(e):
                raise OSError(f"文件 {file_path} 被另一个进程占用。请确保没有其他程序正在访问该文件。") from e
            elif "WinError 123" in str(e):
                raise OSError(f"文件路径 {file_path} 无效。请检查文件路径是否包含非法字符。") from e
            elif "WinError 3" in str(e):
                raise OSError(f"找不到文件路径 {file_path}。请确保文件存在。") from e
            raise OSError(f"读取文件 {file_path} 时发生错误：{str(e)}") from e

    async def add_case(self, case_data: Dict):
        """Add a new case to the database"""
        self.cases.append(case_data)
        await self._save_json(self.cases_file, {"cases": self.cases})

    async def get_all_cases(self) -> List[Dict]:
        """Get all cases from the database"""
        return self.cases

    async def get_all_laws(self) -> Dict:
        """Get all traffic laws"""
        return self.laws

    async def add_law(self, law_category: str, law_data: Dict):
        """Add a new law to the database"""
        if law_category not in self.laws:
            self.laws[law_category] = []
        self.laws[law_category].append(law_data)
        await self._save_json(self.laws_file, self.laws)
        
    async def clear_database(self):
        """Clear all cases from the database"""
        self.cases = []
        await self._save_json(self.cases_file, {"cases": []})
