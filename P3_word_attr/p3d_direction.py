"""
P3-D: 动作方向属性
判断动词的动作方向: 施事→受事, 内向/外向/互向
对构建"主语→动词→宾语"三元关系至关重要
"""
from typing import Dict


class P3D_Direction:
    """动作方向判定"""

    def __init__(self):
        # 外向动作: 主语主动作用于宾语
        self.outward_verbs = {
            "喜欢", "爱", "恨", "讨厌", "帮助", "保护", "照顾", "支持",
            "鼓励", "教", "给", "送", "告诉", "问", "打", "骂", "批评",
            "表扬", "感谢", "赞美", "写", "画", "做", "建", "买", "卖",
            "吃", "喝", "看", "听", "读", "拿", "找", "带", "开", "种",
            "唱", "玩", "学习", "制造", "建造", "修理", "检查", "完成",
            "攻击", "伤害", "欺骗", "背叛", "原谅", "理解", "关心",
        }

        # 内向动作: 主语自身状态变化
        self.inward_verbs = {
            "哭", "笑", "睡", "醒", "坐", "站", "走", "跑", "跳",
            "难过", "开心", "高兴", "伤心", "痛苦", "害怕", "担心",
            "紧张", "生气", "愤怒", "失望", "满意", "满足",
            "死", "活", "病", "累", "饿", "渴", "困",
        }

        # 互向动作: 需要双方参与
        self.mutual_verbs = {
            "结婚", "离婚", "合作", "竞争", "打架", "吵架", "和好",
            "见面", "约会", "聊天", "讨论", "谈判", "交换", "分享",
        }

        # 认知动作: 主语内心活动
        self.cognitive_verbs = {
            "想", "思考", "知道", "认为", "觉得", "相信", "怀疑",
            "忘记", "记得", "理解", "明白", "决定", "计划", "想象",
            "希望", "期待", "打算", "准备",
        }

    def process(self, word: str, context: dict = None) -> Dict:
        """返回: {direction, target_type, confidence}"""
        if word in self.outward_verbs:
            return {"direction": "outward", "target_type": "object", "confidence": 0.90}
        if word in self.inward_verbs:
            return {"direction": "inward", "target_type": "self", "confidence": 0.90}
        if word in self.mutual_verbs:
            return {"direction": "mutual", "target_type": "both", "confidence": 0.90}
        if word in self.cognitive_verbs:
            return {"direction": "cognitive", "target_type": "internal", "confidence": 0.85}

        return {"direction": "unknown", "target_type": "unknown", "confidence": 0.3}
