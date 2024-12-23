import json
import random
from datetime import datetime, timedelta
from typing import List, Dict

class TrafficCaseGenerator:
    def __init__(self):
        self.courts = [
            "北京市第一中级人民法院", "上海市浦东新区人民法院", "广州市天河区人民法院",
            "深圳市南山区人民法院", "杭州市西湖区人民法院", "成都市锦江区人民法院"
        ]
        self.vehicles = ["小型轿车", "货车", "客车", "电动自行车", "摩托车", "SUV"]
        self.weather = ["晴朗", "雨天", "雪天", "大雾", "阴天"]
        self.times = ["早上", "上午", "中午", "下午", "晚上", "凌晨"]
        self.violations = [
            "超速行驶",
            "酒后驾驶",
            "疲劳驾驶",
            "违规超车",
            "闯红灯",
            "逆向行驶",
            "未保持安全距离",
            "违规变道",
            "未礼让行人",
            "驾驶技术不熟练"
        ]
        self.locations = [
            "城市快速路",
            "高速公路",
            "主干道",
            "居民区道路",
            "乡村公路",
            "十字路口",
            "环形路口",
            "隧道内",
            "桥梁上",
            "学校门前道路"
        ]
        self.consequences = [
            "造成一人轻伤",
            "造成两人轻伤",
            "造成三人轻伤",
            "造成一人重伤",
            "造成两车受损",
            "造成多车追尾",
            "造成车辆翻覆",
            "造成行人受伤",
            "造成车辆报废",
            "无人员伤亡"
        ]
        self.laws = [
            "《中华人民共和国道路交通安全法》第二十二条",
            "《中华人民共和国道路交通安全法》第四十二条",
            "《中华人民共和国刑法》第一百三十三条",
            "《中华人民共和国道路交通安全法实施条例》第六十二条",
            "《机动车驾驶证申领和使用规定》第七十二条"
        ]

    def generate_case_number(self, year: int, court_index: int, case_id: int) -> str:
        """生成规范的案号"""
        return f"（{year}）{court_index:02d}刑初{case_id:04d}号"

    def generate_date(self) -> str:
        """生成2023年内的随机日期"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        days_between = (end_date - start_date).days
        random_date = start_date + timedelta(days=random.randint(0, days_between))
        return random_date.strftime("%Y年%m月%d日")

    def generate_blood_alcohol(self) -> float:
        """生成血液酒精含量（mg/100ml）"""
        return round(random.uniform(20, 200), 1)

    def generate_speed(self) -> int:
        """生成超速速度（km/h）"""
        return random.randint(20, 100)

    def generate_case(self, case_id: int) -> Dict[str, str]:
        """生成单个交通事故案例"""
        date = self.generate_date()
        time = random.choice(self.times)
        court = random.choice(self.courts)
        vehicle = random.choice(self.vehicles)
        weather = random.choice(self.weather)
        location = random.choice(self.locations)
        violation = random.choice(self.violations)
        consequence = random.choice(self.consequences)
        applicable_laws = random.sample(self.laws, k=random.randint(1, 3))

        case_number = self.generate_case_number(2023, random.randint(1, 99), case_id)
        
        # 构建完整判决书格式的案例
        content = f"{court}\n刑事判决书\n{case_number}\n\n"
        content += f"案件来源：交通肇事刑事案件\n\n"
        content += f"案件基本情况：\n{date}{time}，被告人驾驶{vehicle}在{location}行驶。当时天气{weather}，"

        # 根据违法行为类型添加具体细节
        if violation == "酒后驾驶":
            blood_alcohol = self.generate_blood_alcohol()
            content += f"被告人{violation}，血液中酒精含量为{blood_alcohol}mg/100ml，"
        elif violation == "超速行驶":
            speed = self.generate_speed()
            content += f"被告人{violation}，超出限速{speed}公里/小时，"
        else:
            content += f"被告人{violation}，"

        # 添加事故后果和法律依据
        content += f"发生交通事故，{consequence}。\n\n"
        content += "法律依据：\n"
        for law in applicable_laws:
            content += f"{law}\n"

        return {
            "id": str(case_id),
            "content": content,
            "case_number": case_number,
            "court": court,
            "applicable_laws": applicable_laws
        }

    def generate_cases(self, start_id: int, count: int) -> List[Dict[str, str]]:
        """生成指定数量的交通事故案例"""
        return [self.generate_case(i) for i in range(start_id, start_id + count)]

def main():
    try:
        # 尝试读取现有案例
        with open("data/cases.json", "r", encoding="utf-8") as f:
            existing_cases = json.load(f)
        existing_count = len(existing_cases)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或格式错误，从头开始生成
        existing_cases = []
        existing_count = 0
    
    # 计算需要生成的案例数量
    needed_count = 500 - existing_count
    
    # 生成新案例
    generator = TrafficCaseGenerator()
    new_cases = generator.generate_cases(existing_count + 1, needed_count)
    
    # 合并现有案例和新案例
    all_cases = existing_cases + new_cases
    
    # 保存所有案例
    with open("data/cases.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully generated {needed_count} new cases. Total cases: {len(all_cases)}")

if __name__ == "__main__":
    main()
