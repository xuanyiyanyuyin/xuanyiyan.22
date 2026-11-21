"""
P3-Role: 人称属性
仅判断词的人称和数 — 不绑定USER/SELF (留给ContextCache)
我=第一人称单数, 你=第二人称单/复数, 他/她=第三人称单数
"""
from typing import Dict


class P3R_PersonRole:
    """人称属性判定 — 只判断语法人称, 不判断对话角色"""

    def __init__(self):
        self.person_map = {
            # 第一人称
            "我":   ("first", "singular"),
            "我们": ("first", "plural"),
            "咱":   ("first", "singular"),
            "咱们": ("first", "plural"),
            # 第二人称
            "你":   ("second", "singular"),
            "你们": ("second", "plural"),
            "您":   ("second", "singular"),  # 敬称, grammatically second person
            # 第三人称
            "他":   ("third", "singular"),
            "她":   ("third", "singular"),
            "它":   ("third", "singular"),
            "他们": ("third", "plural"),
            "她们": ("third", "plural"),
            "它们": ("third", "plural"),
            # 反身/泛指
            "自己": ("reflexive", "singular"),
            "大家": ("general", "plural"),
            "别人": ("general", "singular"),
            "谁":   ("interrogative", "singular"),
        }

    def process(self, word: str, context: dict = None) -> Dict:
        """返回: {person, number, confidence}
        不包含 speaker_role (USER/SELF) — 那由ContextCache决定"""
        if word in self.person_map:
            person, number = self.person_map[word]
            return {
                "person": person,
                "number": number,
                "confidence": 0.99,
            }
        return {
            "person": "none",
            "number": "none",
            "confidence": 0.0,
        }
