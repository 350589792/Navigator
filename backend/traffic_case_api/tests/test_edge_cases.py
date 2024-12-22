import pytest
from fastapi.testclient import TestClient
from app.models import Case
from app.data_manager import CaseDataManager

async def test_empty_database(test_client):
    """测试空数据库情况"""
    # 清空数据库
    data_manager = CaseDataManager()
    await data_manager.clear_database()
    
    response = test_client.post(
        "/analyze_case",
        json={"content": "驾驶员在高速公路超速行驶"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "No cases found in database"

def test_invalid_input(test_client):
    """测试无效输入"""
    # 测试空内容
    response = test_client.post(
        "/analyze_case",
        json={"content": ""}
    )
    assert response.status_code == 422
    
    # 测试缺少必要字段
    response = test_client.post(
        "/analyze_case",
        json={}
    )
    assert response.status_code == 422

def test_multiple_violations(test_client, sample_cases):
    """测试多重违法行为案例"""
    # 包含超速、酒驾和逃逸的复杂案例
    complex_case = {
        "content": "驾驶员饮酒后驾驶机动车，以150公里/小时的速度行驶，撞向路边护栏后逃逸。"
    }
    
    response = test_client.post(
        "/analyze_case",
        json=complex_case
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "similar_cases" in data
    assert len(data["similar_cases"]) > 0
    
    # 验证法条包含所有相关违法行为的法条
    expected_laws = [
        "《中华人民共和国道路交通安全法》第一百一十九条",
        "《中华人民共和国刑法》第一百三十三条",
        "《中华人民共和国刑法》第一百三十三条之一"
    ]
    assert all(law in data["relevant_laws"] for law in expected_laws)
    
    # 验证相似案例包含至少一个相关违法行为
    violations = ["超速", "酒后", "逃逸"]
    for case in data["similar_cases"]:
        assert any(violation in case["content"] for violation in violations)
