import re
import jieba
from typing import List, Dict

# 定义各类交通事故特征关键词
ACCIDENT_FEATURES = {
    "SPEEDING": ["超速", "超速行驶", "高速行驶", "超出限速", "未减速"],
    "HIT_AND_RUN": ["肇事逃逸", "逃逸", "驾车逃逸", "驾驶逃逸", "逃离现场"],
    "TRAFFIC_SIGNAL": ["闯红灯", "闯信号灯", "违反信号灯", "未按信号灯"],
    "VEHICLE_MALFUNCTION": ["车辆故障", "制动失灵", "刹车失灵", "方向失控", "轮胎爆裂"],
    "PEDESTRIAN": ["行人事故", "撞击行人", "碾压行人", "行人冲突", "人行横道"],
    "SEVERITY": ["重伤", "死亡", "轻伤", "残疾", "骨折", "昏迷"],
    "LOCATION": ["高速公路", "城市道路", "交叉路口", "人行横道", "隧道", "桥梁"],
    "WEATHER": ["雨天", "雾天", "雪天", "湿滑", "结冰", "大风"],
    "VEHICLE_TYPE": ["客车", "货车", "摩托车", "电动车", "自行车", "三轮车"],
    "VIOLATION": ["违章", "违规", "违反规定", "违反交规", "未按规定"]
}

def extract_numeric_values(text: str) -> str:
    """
    提取并标记数值信息（速度、距离等）
    Args:
        text: 输入文本
    Returns:
        处理后的文本，添加数值相关标记
    """
    # 速度模式匹配（km/h，公里/小时等）
    speed_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:km/h|公里/小时|千米/小时)',
        r'时速\s*(\d+(?:\.\d+)?)\s*(?:km|公里|千米)',
        r'以\s*(\d+(?:\.\d+)?)\s*(?:公里/小时|千米/小时)',  # 新增模式
        r'(\d+(?:\.\d+)?)\s*码表显示'
    ]
    
    # 距离模式匹配（米，公里等）
    distance_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:米|m)',
        r'(\d+(?:\.\d+)?)\s*(?:公里|km|千米)',
        r'相距\s*(\d+(?:\.\d+)?)\s*(?:米|m|公里|km)'
    ]
    
    enhanced_text = text
    
    # 处理速度
    for pattern in speed_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            speed = float(match.group(1))
            # 根据《道路交通安全法》的速度分级添加标记
            if speed > 120:
                enhanced_text += f" SPEED_EXTREME_{int(speed)} SPEED_EXTREME_{int(speed)}"
            elif speed > 80:
                enhanced_text += f" SPEED_HIGH_{int(speed)} SPEED_HIGH_{int(speed)}"
            else:
                enhanced_text += f" SPEED_{int(speed)} SPEED_{int(speed)}"
    
    # 处理距离
    for pattern in distance_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            distance = float(match.group(1))
            enhanced_text += f" DISTANCE_{int(distance)} DISTANCE_{int(distance)}"
    
    return enhanced_text

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
    matches = re.finditer(pattern, text)
    enhanced_text = text
    for match in matches:
        value = float(match.group(1))
        # 根据《道路交通安全法》，80mg/100ml为醉驾标准
        if value >= 80:
            enhanced_text += f" BAC_HIGH_{value} BAC_HIGH BAC_HIGH"  # 添加具体数值和分类标记
        else:
            enhanced_text += f" BAC_LOW_{value} BAC_LOW BAC_LOW"
    return enhanced_text

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

def add_feature_tokens(text: str) -> str:
    """
    检测交通事故特征关键词并添加对应标记
    Args:
        text: 输入文本
    Returns:
        处理后的文本，添加各类特征标记
    """
    enhanced_text = text
    for feature_type, keywords in ACCIDENT_FEATURES.items():
        if any(keyword in text for keyword in keywords):
            # 重复添加特征标记以增加权重
            enhanced_text += f" {feature_type} {feature_type} {feature_type}"
    return enhanced_text

def preprocess_text(text: str) -> str:
    """对中文文本进行预处理，使用自定义分词器"""
    return " ".join(custom_tokenizer(text))

def custom_tokenizer(text: str) -> List[str]:
    """
    自定义分词器，整合所有特征检测
    Args:
        text: 输入文本
    Returns:
        分词后的词列表
    """
    # 添加特定标记
    enhanced_text = text
    enhanced_text = extract_numeric_values(enhanced_text)  # 首先提取数值信息
    enhanced_text = add_bac_token(enhanced_text)
    enhanced_text = add_drunk_tokens(enhanced_text)
    enhanced_text = add_feature_tokens(enhanced_text)
    
    # 提取所有特征标记（在分词前）
    feature_tokens = []
    for token in enhanced_text.split():
        if any(keyword in token for keyword in ["SPEED", "BAC", "DRUNK", "DISTANCE"]):
            feature_tokens.append(token)
    
    # 使用结巴分词处理原始文本
    base_tokens = jieba.lcut(text)
    
    # 合并基本分词和特征标记
    return base_tokens + feature_tokens
