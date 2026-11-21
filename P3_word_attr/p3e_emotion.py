"""
P3-E: 情感极性+强度属性
判断词的情感倾向: 积极/消极/中性 + 强度(0~1)
覆盖情感动词、形容词、副词的情感色彩
"""
from typing import Dict, List, Tuple


class P3E_Emotion:
    """情感极性判定 — 基于词表 + 强度分级"""

    def __init__(self):
        # 情感词表 (词 → (极性, 强度, 类别))
        self.emotion_words = {
            # 积极-爱意 (高强度)
            "爱":   ("positive", 1.0, "affection"),
            "喜欢": ("positive", 0.85, "affection"),
            "热爱": ("positive", 1.0, "affection"),
            "疼爱": ("positive", 0.95, "affection"),
            "宠爱": ("positive", 0.9, "affection"),
            # 积极-开心
            "开心": ("positive", 0.8, "joy"),
            "高兴": ("positive", 0.8, "joy"),
            "快乐": ("positive", 0.85, "joy"),
            "幸福": ("positive", 1.0, "joy"),
            "愉快": ("positive", 0.75, "joy"),
            "满意": ("positive", 0.7, "satisfaction"),
            "满足": ("positive", 0.7, "satisfaction"),
            # 积极-赞美
            "好":   ("positive", 0.5, "praise"),
            "棒":   ("positive", 0.7, "praise"),
            "优秀": ("positive", 0.8, "praise"),
            "厉害": ("positive", 0.8, "praise"),
            "美":   ("positive", 0.6, "praise"),
            "漂亮": ("positive", 0.7, "praise"),
            # 积极-希望
            "希望": ("positive", 0.7, "hope"),
            "期待": ("positive", 0.75, "hope"),
            "相信": ("positive", 0.65, "trust"),
            "信任": ("positive", 0.7, "trust"),
            # 积极-帮助
            "帮助": ("positive", 0.7, "help"),
            "支持": ("positive", 0.7, "support"),
            "鼓励": ("positive", 0.8, "encourage"),
            "保护": ("positive", 0.75, "protect"),
            "照顾": ("positive", 0.7, "care"),
            # 积极-副词
            "也":   ("positive", 0.3, "agreement"),
            "还":   ("positive", 0.2, "continuation"),
            "很":   ("positive", 0.4, "intensifier"),
            "非常": ("positive", 0.6, "intensifier"),
            # 消极-讨厌
            "讨厌": ("negative", 0.8, "dislike"),
            "恨":   ("negative", 1.0, "hatred"),
            "烦":   ("negative", 0.6, "annoyance"),
            "厌恶": ("negative", 0.85, "disgust"),
            # 消极-悲伤
            "难过": ("negative", 0.75, "sadness"),
            "伤心": ("negative", 0.8, "sadness"),
            "痛苦": ("negative", 0.9, "pain"),
            "哭":   ("negative", 0.7, "cry"),
            "失望": ("negative", 0.7, "disappointment"),
            # 消极-愤怒
            "生气": ("negative", 0.7, "anger"),
            "愤怒": ("negative", 0.85, "anger"),
            "恼火": ("negative", 0.75, "anger"),
            # 消极-恐惧
            "害怕": ("negative", 0.7, "fear"),
            "担心": ("negative", 0.6, "worry"),
            "紧张": ("negative", 0.5, "anxiety"),
            # 消极-否定
            "不":   ("negative", 0.4, "negation"),
            "没":   ("negative", 0.3, "negation"),
            "别":   ("negative", 0.4, "prohibition"),
            # 中性
            "知道": ("neutral", 0.0, "knowledge"),  # 中性词, 低置信度
            "认为": ("neutral", 0.0, "opinion"),
            "觉得": ("neutral", 0.0, "opinion"),
            "想":   ("neutral", 0.0, "think"),
            "说":   ("neutral", 0.0, "speak"),
            "问":   ("neutral", 0.0, "ask"),
            "看":   ("neutral", 0.0, "observe"),
            "听":   ("neutral", 0.0, "listen"),
            "去":   ("neutral", 0.0, "go"),
            "来":   ("neutral", 0.0, "come"),
            "是":   ("neutral", 0.0, "be"),
            "有":   ("neutral", 0.0, "have"),
        }

    def process(self, word: str, context: dict = None) -> Dict:
        """返回: {polarity, intensity, category, confidence}"""
        if word in self.emotion_words:
            pol, intensity, cat = self.emotion_words[word]
            return {
                "polarity": pol,
                "intensity": intensity,
                "category": cat,
                "confidence": 0.95,
            }
        return {
            "polarity": "neutral",
            "intensity": 0.0,
            "category": "none",
            "confidence": 0.5,
        }
