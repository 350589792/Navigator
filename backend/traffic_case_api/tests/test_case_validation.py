import json
import pytest
from pathlib import Path
import re

def load_cases():
    data_path = Path("data/cases.json")
    assert data_path.exists(), "Cases file not found"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_case_count():
    """验证案例总数是否为500"""
    cases = load_cases()
    assert len(cases) == 500, f"Expected 500 cases, got {len(cases)}"

def test_unique_ids():
    """验证ID唯一性"""
    cases = load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"

def test_required_fields():
    """验证必需字段存在性"""
    cases = load_cases()
    required_fields = ["id", "content", "case_number", "court", "applicable_laws"]
    for case in cases:
        for field in required_fields:
            assert field in case, f"Missing required field: {field}"

def test_case_number_format():
    """验证案号格式"""
    cases = load_cases()
    pattern = r"（\d{4}）\d{2}刑初\d{4}号"
    for case in cases:
        assert "case_number" in case, "Case number missing"
        assert re.match(pattern, case["case_number"]), f"Invalid case number format: {case['case_number']}"

def test_court_names():
    """验证法院名称"""
    cases = load_cases()
    court_pattern = r".*人民法院$"
    for case in cases:
        assert "court" in case, "Court name missing"
        assert re.match(court_pattern, case["court"]), f"Invalid court name: {case['court']}"

def test_content_completeness():
    """验证判决内容完整性"""
    cases = load_cases()
    required_content = ["案件来源", "案件基本情况", "法律依据"]
    for case in cases:
        content = case["content"]
        for required in required_content:
            assert required in content, f"Missing required content section: {required}"

def test_legal_citations():
    """验证法条引用"""
    cases = load_cases()
    law_pattern = r"《.*?》第.*?条"
    for case in cases:
        assert "applicable_laws" in case, "Legal citations missing"
        for law in case["applicable_laws"]:
            assert re.match(law_pattern, law), f"Invalid legal citation format: {law}"

def test_content_length():
    """验证内容长度合理性"""
    cases = load_cases()
    for case in cases:
        content_length = len(case["content"])
        assert 100 <= content_length <= 2000, f"Content length ({content_length}) out of reasonable range"

def test_date_format():
    """验证日期格式"""
    cases = load_cases()
    date_pattern = r"\d{4}年\d{1,2}月\d{1,2}日"
    for case in cases:
        content = case["content"]
        assert re.search(date_pattern, content), f"Invalid or missing date format in content"
