import asyncio
from app.data_manager import CaseDataManager

async def init_database():
    data_manager = CaseDataManager()
    sample_cases = [
        {
            "content": "2023年7月10日晚上9点，王某驾驶小型轿车在城市快速路上行驶。当时天气晴朗，王某超速行驶且未保持安全距离，与前方减速的车辆发生追尾。事故造成两人轻伤，车辆受损。经检测，王某血液中酒精含量为90mg/100ml。",
            "id": "1"
        },
        {
            "content": "2023年6月5日下午3点，李某驾驶小型轿车在雨天路滑的情况下，未减速行驶，与前方等红灯的车辆发生追尾。事故造成一人轻伤，两车受损。",
            "id": "2"
        },
        {
            "content": "2023年8月20日凌晨2点，张某酒后驾驶机动车，血液中酒精含量为98mg/100ml，在转弯处与对向来车相撞，造成两车损坏，无人员伤亡。",
            "id": "3"
        },
        {
            "content": "2023年5月15日早上7点，赵某驾驶货车在高速公路上行驶时，因疲劳驾驶操作失误，与前方车辆发生追尾。事故造成三人受伤，车辆严重受损。",
            "id": "4"
        },
        {
            "content": "2023年9月1日中午12点，陈某驾驶小型轿车，在超车过程中未保持安全距离，与旁边车道的车辆发生碰撞。事故造成两车受损，一人轻伤。",
            "id": "5"
        }
    ]
    
    await data_manager.save_cases(sample_cases)
    print("Database initialized with sample cases.")

if __name__ == "__main__":
    asyncio.run(init_database())
