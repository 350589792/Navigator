"""数值处理工具模块"""
import re
from typing import List

def extract_speed(text: str) -> List[str]:
    """
    提取文本中的速度相关信息
    
    Args:
        text: 输入文本
        
    Returns:
        速度相关标记列表
    """
    speed_tokens = []
    # 匹配速度模式
    speed_pattern = r'(\d+)(?:公里/小时|km/h|千米/小时)'
    matches = re.finditer(speed_pattern, text)
    
    for match in matches:
        speed = int(match.group(1))
        if speed > 120:
            speed_tokens.append('SPEED_VERY_HIGH')
        elif speed > 80:
            speed_tokens.append('SPEED_HIGH')
    
    return speed_tokens

def add_numeric_tokens(text: str) -> str:
    """
    添加数值相关标记
    
    Args:
        text: 输入文本
        
    Returns:
        添加数值标记后的文本
    """
    # 添加速度标记
    speed_tokens = extract_speed(text)
    if speed_tokens:
        text = f"{text} {' '.join(speed_tokens)}"
    
    return text
