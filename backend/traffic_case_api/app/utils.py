import re
import jieba
from typing import List

def add_bac_token(text: str) -> str:
    """
    检测血液酒精含量并添加特定标记
    Args:
        text: 输入文本
    Returns:
        处理后的文本，如果检测到高血液酒精含量则添加BAC_HIGH标记
    """
    # 匹配血液酒精含量数值，如135.6mg/100ml
    pattern = r'(\d+(?:\.\d+)?)\s*mg/100ml'
    match = re.search(pattern, text)
    if match:
        value = float(match.group(1))
        # 根据《道路交通安全法》，80mg/100ml为醉驾标准
        if value >= 80:
            text += " BAC_HIGH BAC_HIGH BAC_HIGH"  # 重复添加以增加权重
    return text

def add_drunk_tokens(text: str) -> str:
    """
    检测酒驾相关关键词并添加特定标记
    Args:
        text: 输入文本
    Returns:
        处理后的文本，添加酒驾相关标记
    """
    keywords = ["酒后驾驶", "酒驾", "醉酒驾驶", "饮酒驾驶"]
    if any(keyword in text for keyword in keywords):
        text += " DRUNK_DRIVING DRUNK_DRIVING DRUNK_DRIVING"  # 重复添加以增加权重
    return text

def custom_tokenizer(text: str) -> List[str]:
    """
    自定义分词器，整合血液酒精含量和酒驾检测
    Args:
        text: 输入文本
    Returns:
        分词后的词列表
    """
    # 添加特定标记
    text = add_bac_token(text)
    text = add_drunk_tokens(text)
    # 使用结巴分词
    return jieba.lcut(text)
