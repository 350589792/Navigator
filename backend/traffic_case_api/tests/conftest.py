import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.data_manager import CaseDataManager
from app.scraper import TrafficCaseScraper
from app.models import CaseInput, CaseAnalysisResponse

@pytest_asyncio.fixture(scope="function")
async def test_data():
    """初始化测试数据"""
    # Create test cases
    test_cases = [
        {
            "case_number": "TEST-2024-001",
            "court_name": "测试法院",
            "content": "驾驶员血液酒精含量为98.5mg/100ml，以120公里/小时的速度行驶",
            "judgment_date": "2024-01-15",
            "similarity_score": 0.85
        },
        {
            "case_number": "TEST-2024-002",
            "court_name": "测试法院",
            "content": "驾驶员血液酒精含量为105mg/100ml，闯红灯并超速行驶",
            "judgment_date": "2024-01-16",
            "similarity_score": 0.92
        }
    ]
    
    # Create test laws
    test_laws = {
        "traffic": [
            {
                "law_name": "道路交通安全法",
                "article_number": "第91条",
                "content": "饮酒后驾驶机动车的，处暂扣6个月机动车驾驶证"
            }
        ]
    }
    
    return {"cases": test_cases, "laws": test_laws}

@pytest.fixture(scope="function")
def test_client(test_data):
    """创建测试用的FastAPI客户端"""
    import asyncio
    
    # Initialize test data in app state
    async def setup():
        app.state.data_manager = await CaseDataManager.create()
        for case in test_data["cases"]:
            await app.state.data_manager.add_case(case)
        for law in test_data["laws"]["traffic"]:
            await app.state.data_manager.add_law("traffic", law)
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup())
    
    with TestClient(app) as client:
        yield client
    
    # Cleanup
    async def cleanup():
        if hasattr(app.state, 'data_manager') and app.state.data_manager is not None:
            await app.state.data_manager.clear_database()
            app.state.data_manager = None
        if hasattr(app.state, 'case_scraper'):
            app.state.case_scraper = None
    
    loop.run_until_complete(cleanup())
