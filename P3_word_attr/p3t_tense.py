"""
P3-T: 时态/体态属性检测 — P3第9子模块
========================================
中文时态不需动词变位, 靠助词和副词表达。
纯规则引擎, 0参数。
"""
from typing import Dict, Optional


class P3T_TenseAspect:
    """时态/体态属性检测器"""

    # 时态标记词
    TENSE_MARKERS = {
        # 完成体 (perfective)
        "了": ("completive", 0.9),
        "过": ("experiential", 0.9),
        "完": ("completive", 0.7),
        "好": ("resultative", 0.6),
        "掉": ("disposal_completive", 0.7),
        "到": ("achievement", 0.6),
        # 进行体 (progressive)
        "在": ("progressive", 0.7),
        "正在": ("progressive", 0.95),
        "着": ("durative", 0.9),
        "下去": ("continuative", 0.8),
        # 起始体 (inchoative)
        "起来": ("inchoative", 0.8),
        "开始": ("inchoative", 0.85),
        "起": ("inchoative", 0.6),
        # 将来/未然 (future/irrealis)
        "将": ("future", 0.9),
        "将要": ("future", 0.95),
        "会": ("future", 0.8),
        "要": ("future", 0.7),
        "就要": ("imminent", 0.9),
        "快要": ("imminent", 0.85),
        "即将": ("imminent", 0.95),
        # 已然/曾然 (already/once)
        "已经": ("already", 0.95),
        "已": ("already", 0.9),
        "曾经": ("once", 0.9),
        "曾": ("once", 0.85),
        "早就": ("already_early", 0.9),
        # 未然 (not_yet)
        "还没": ("not_yet", 0.9),
        "还未": ("not_yet", 0.95),
        "没": ("not_yet", 0.7),
        "未": ("not_yet", 0.8),
        # 持续/反复 (iterative)
        "一直": ("continuous", 0.9),
        "总是": ("habitual", 0.85),
        "经常": ("frequentative", 0.8),
        "常常": ("frequentative", 0.85),
        "往往": ("habitual", 0.8),
        "老": ("frequentative", 0.6),
        "又": ("iterative", 0.7),
        "再": ("iterative", 0.8),
        "还": ("persistive", 0.7),
        "仍然": ("persistive", 0.9),
        "依然": ("persistive", 0.9),
        "依旧": ("persistive", 0.85),
    }

    # 时间名词
    TIME_NOUNS = {
        "今天", "昨天", "明天", "前天", "后天",
        "现在", "刚才", "刚刚", "马上", "立刻", "立即",
        "以前", "以前", "以后", "后来", "将来", "未来",
        "今年", "去年", "明年", "本月", "上月", "下月",
        "今天", "今晚", "今早", "今晨",
        "上午", "下午", "晚上", "早晨", "傍晚", "深夜",
        "周一", "周二", "周三", "周四", "周五", "周六", "周日",
        "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期天",
        "春天", "夏天", "秋天", "冬天", "春季", "夏季", "秋季", "冬季",
        "小时候", "年轻时", "老年", "晚年",
        "古代", "现代", "当代", "近代", "远古",
        "最初", "最后", "最终", "起初", "当初",
    }

    def process(self, word: str, context: Optional[dict] = None) -> dict:
        result = {
            "has_tense": False,
            "tense_type": "",      # past/present/future/timeless
            "aspect_type": "",     # completive/progressive/inchoative/durative/habitual/experiential/imminent
            "time_reference": "",  # specific_time/relative_time/vague/general
            "tense_confidence": 0.0,
            "is_time_noun": False,
        }

        # 检查时态标记词
        if word in self.TENSE_MARKERS:
            aspect, conf = self.TENSE_MARKERS[word]
            result["has_tense"] = True
            result["aspect_type"] = aspect
            result["tense_confidence"] = conf

            # 映射aspect到tense
            past_aspects = {"completive", "experiential", "once", "already", "already_early", "disposal_completive"}
            present_aspects = {"progressive", "durative", "persistive", "continuous", "habitual", "frequentative"}
            future_aspects = {"future", "imminent", "inchoative"}
            if aspect in past_aspects:
                result["tense_type"] = "past"
            elif aspect in present_aspects:
                result["tense_type"] = "present"
            elif aspect in future_aspects:
                result["tense_type"] = "future"
            elif aspect == "not_yet":
                result["tense_type"] = "future"
            else:
                result["tense_type"] = "timeless"

        # 检查时间名词
        if word in self.TIME_NOUNS:
            result["has_tense"] = True
            result["time_reference"] = "specific_time"
            result["is_time_noun"] = True
            result["tense_confidence"] = max(result["tense_confidence"], 0.85)

        return result

    def process_sentence(self, words: list) -> list:
        return [self.process(w) for w in words]

    def sentence_tense(self, words: list) -> str:
        """推断句子整体时态"""
        results = self.process_sentence(words)
        tenses = {}
        for r in results:
            t = r.get("tense_type", "")
            if t and t != "timeless":
                tenses[t] = tenses.get(t, 0) + r.get("tense_confidence", 0)
        if not tenses:
            return "timeless"
        return max(tenses, key=tenses.get)


if __name__ == "__main__":
    p3t = P3T_TenseAspect()
    tests = [
        ["我", "已经", "完成", "了", "训练"],
        ["他", "正在", "跑", "着", "模型"],
        ["明天", "将", "开始", "新", "实验"],
        ["我们", "曾经", "尝试", "过", "这", "方法"],
        ["GPU", "还", "在", "计算"],
        ["损失", "还没", "下降"],
        ["今天下午", "要", "调参"],
    ]
    for words in tests:
        results = p3t.process_sentence(words)
        tense = p3t.sentence_tense(words)
        marks = [(w, r["aspect_type"]) for w, r in zip(words, results) if r["has_tense"]]
        print(f"'{''.join(words)}' → 时态={tense} 标记={marks}")
