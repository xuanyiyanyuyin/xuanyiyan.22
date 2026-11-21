"""
P3-Q: 疑问槽属性
判断疑问词类型 → 映射到答案槽位
谁=主语疑问(人物槽), 什么=宾语疑问(对象槽), 哪里=地点疑问, 怎么=方式疑问
"""
from typing import Dict


class P3Q_QuestionSlot:
    """疑问槽判定 — 疑问词→槽位映射"""

    def __init__(self):
        # 疑问词→槽位映射表
        self.question_map = {
            "谁":   ("subject", "person", 0.99),
            "什么": ("object", "thing", 0.99),
            "哪里": ("place", "location", 0.99),
            "哪儿": ("place", "location", 0.99),
            "怎么": ("manner", "method", 0.99),
            "怎样": ("manner", "method", 0.99),
            "如何": ("manner", "method", 0.95),
            "为什么": ("reason", "cause", 0.99),
            "哪个": ("subject_object", "selection", 0.90),
            "哪些": ("subject_object", "selection", 0.90),
            "什么时候": ("time", "time_point", 0.99),
            "多久": ("time_duration", "duration", 0.99),
            "多少": ("quantity", "number", 0.99),
            "多远": ("distance", "distance", 0.99),
            "多大": ("size_age", "measure", 0.95),
            "怎么样": ("manner_state", "description", 0.99),
        }

        # 槽位→预期答案属性
        self.slot_to_answer_attr = {
            "subject":        {"target_attr": "主语", "semantic": "人物"},
            "object":         {"target_attr": "宾语", "semantic": None},
            "place":          {"target_attr": "宾语", "semantic": "地点"},
            "manner":         {"target_attr": "状语", "semantic": None},
            "reason":         {"target_attr": "状语", "semantic": None},
            "time":           {"target_attr": "状语", "semantic": None},
            "subject_object": {"target_attr": "主语/宾语", "semantic": None},
        }

    def process(self, word: str, context: dict = None) -> Dict:
        """返回: {question_slot, slot_type, target_attr, target_semantic, confidence}"""
        if word in self.question_map:
            slot, slot_type, conf = self.question_map[word]
            target = self.slot_to_answer_attr.get(slot, {})
            return {
                "question_slot": slot,
                "slot_type": slot_type,
                "target_attr": target.get("target_attr", ""),
                "target_semantic": target.get("semantic", ""),
                "confidence": conf,
            }
        return {
            "question_slot": "none",
            "slot_type": "",
            "target_attr": "",
            "target_semantic": "",
            "confidence": 0.0,
        }
