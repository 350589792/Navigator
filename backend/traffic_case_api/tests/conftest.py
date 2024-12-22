import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_manager import CaseDataManager

@pytest.fixture
async def test_data():
    """测试数据"""
    return [
        {
            "content": "驾驶员甲在高速公路上以120公里/小时的速度行驶，造成追尾事故。",
            "title": "超速案例1",
            "court": "某市交通法院",
            "case_number": "2023-001",
            "judgment_date": "2023-01-01"
        },
        {
            "content": "驾驶员乙酒后驾驶机动车，血液中酒精含量为80mg/100ml，与路边护栏相撞。",
            "title": "酒驾案例1",
            "court": "某市交通法院",
            "case_number": "2023-002",
            "judgment_date": "2023-01-02"
        },
        {
            "content": "驾驶员丙闯红灯，与一辆正常行驶的车辆相撞后逃逸。",
            "title": "闯红灯及逃逸案例1",
            "court": "某市交通法院",
            "case_number": "2023-003",
            "judgment_date": "2023-01-03"
        },
        {
            "content": "驾驶员丁酒后驾驶，血液中酒精含量为90mg/100ml，以140公里/小时的速度行驶导致追尾。",
            "title": "酒驾超速案例",
            "court": "某市交通法院",
            "case_number": "2023-004",
            "judgment_date": "2023-01-04"
        },
        {
            "content": "驾驶员戊饮酒后驾驶机动车，以150公里/小时的速度行驶，撞向路边护栏后逃逸。",
            "title": "酒驾超速逃逸案例",
            "court": "某市交通法院",
            "case_number": "2023-005",
            "judgment_date": "2023-01-05"
        }
    ]

@pytest.fixture
async def test_client(test_data):
    """创建测试客户端"""
    # 初始化数据管理器
    data_manager = CaseDataManager()
    await data_manager.clear_database()
    await data_manager.save_cases(test_data)
    
    # 创建测试客户端
    client = TestClient(app)
    
    yield client
    
    # 清理测试数据
    await data_manager.clear_database()

@pytest.fixture
def sample_cases():
    """样例案例数据"""
    return {
        "speeding": {
            "content": "驾驶员在高速公路上以150公里/小时的速度行驶，因车速过快导致追尾事故。",
            "expected_laws": ["《中华人民共和国道路交通安全法》第一百一十九条"]
        },
        "drunk_driving": {
            "content": "驾驶员饮酒后驾驶机动车，血液中酒精含量为98mg/100ml。",
            "expected_laws": [
                "《中华人民共和国道路交通安全法》第一百一十九条",
                "《中华人民共和国刑法》第一百三十三条"
            ]
        },
        "hit_and_run": {
            "content": "驾驶员闯红灯撞向行人后驾车逃逸。",
            "expected_laws": [
                "《中华人民共和国道路交通安全法》第一百一十九条",
                "《中华人民共和国刑法》第一百三十三条之一"
            ]
        },
        "multiple_violations": {
            "content": "驾驶员饮酒后驾驶机动车，以150公里/小时的速度行驶，撞向路边护栏后逃逸。",
            "expected_laws": [
                "《中华人民共和国道路交通安全法》第一百一十九条",
                "《中华人民共和国刑法》第一百三十三条",
                "《中华人民共和国刑法》第一百三十三条之一"
            ]
        }
    }
