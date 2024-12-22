from typing import List, Optional
import re
import jieba

# 关键词模式列表，包含各种交通违规和事故特征
KEYWORD_PATTERNS = [
    # 超速违规
    "超速", "超速行驶", "高速", "时速", "公里/小时", "km/h",
    # 肇事逃逸
    "肇事逃逸", "逃逸", "逃离现场", "驾驶后逃跑",
    # 交通信号违规
    "闯红灯", "闯黄灯", "违反交通信号", "信号灯",
    # 车辆故障
    "车辆故障", "制动失灵", "刹车失灵", "轮胎爆胎", "方向失控",
    # 行人事故
    "行人", "撞击行人", "人行横道", "斑马线", "过马路",
    # 伤亡程度
    "重伤", "死亡", "轻伤", "致人死亡", "致人重伤",
    # 道路条件
    "高速公路", "国道", "省道", "乡道", "路口", "十字路口",
    # 天气条件
    "雨天", "雨滑", "大雨", "暴雨", "雾天", "能见度低",
    # 酒驾毒驾
    "酒后", "醉酒", "饮酒", "毒驾", "吸毒",
    # 其他危险驾驶
    "疲劳驾驶", "分心驾驶", "注意力不集中", "违章超车"
]

def add_feature_tokens(text: str) -> str:
    """
    检测文本中的关键特征并添加重复标记以增加其权重
    
    Args:
        text: 输入的案例文本
    
    Returns:
        添加了特征标记的文本
    """
    # 检查每个关键词模式
    for keyword in KEYWORD_PATTERNS:
        if keyword in text:
            # 为每个找到的特征添加三次重复标记以增加其权重
            text += f" FEATURE_{keyword} FEATURE_{keyword} FEATURE_{keyword}"
    
    return text

def custom_tokenizer(text: str) -> List[str]:
    """
    自定义分词器，集成特征检测和标记增强
    
    Args:
        text: 输入的案例文本
    
    Returns:
        分词后的标记列表
    """
    # 首先添加特征标记
    text = add_feature_tokens(text)
    
    # 添加数值相关标记
    from .numeric_utils import add_numeric_tokens
    text = add_numeric_tokens(text)
    
    # 使用结巴分词进行基础分词
    tokens = list(jieba.cut(text))
    
    return tokens
