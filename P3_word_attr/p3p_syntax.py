"""
P3-P: 句法位置候选属性
判断词可能处于什么句法位置: 主语/谓语/宾语/状语/补语/定语
可多标签, 带置信度
"""
from typing import Dict, List, Tuple


class P3P_SyntaxPosition:
    """句法位置候选 — 基于属性词表 + 位置启发式"""

    def __init__(self):
        # 从现有P3属性族加载
        self.subj_words = set()
        self.verb_words = set()
        self.obj_words = set()
        self.adj_words = set()
        self.adv_words = set()
        self.comp_words = set()
        self.func_words = set()

        import os
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "P3_data")
        file_map = {
            "subj": (self.subj_words, "主语"),
            "verb": (self.verb_words, "谓语"),
            "obj": (self.obj_words, "宾语"),
            "adj": (self.adj_words, "定语"),
            "adv": (self.adv_words, "状语"),
            "comp": (self.comp_words, "补语"),
            "func": (self.func_words, "虚词"),
        }
        for fname, (word_set, _) in file_map.items():
            path = os.path.join(data_dir, f"words_{fname}_clean.txt")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for w in f:
                        w = w.strip()
                        if w:
                            word_set.add(w)

        # 高频动词补丁: 这些词必须被识别为谓语候选
        # (原始P3数据中部分高频动词缺失, 显式补充)
        self.essential_verbs = {
            "喜欢", "去", "帮助", "建", "建造", "制造", "完成", "画", "唱",
            "送", "拿", "用", "读", "种", "开", "找", "带", "回", "来", "到",
            "跑", "走", "说", "问", "想", "知道", "认为", "觉得", "给", "让",
            "开始", "结束", "使用", "相信", "拥有", "检查", "操作", "阅读",
            "修理", "跑步", "走路", "说话", "做饭", "写字", "工作",
        }

        # 人称代词: 通常是主语, 有时宾语
        self.pronouns = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们", "谁"}

    def process(self, word: str, context: dict = None) -> Dict:
        """返回: {syntax_candidates: [(role, confidence), ...]}"""
        results = []

        # 规则1: 在属性词表中 (谓语族置信度优先)
        # 动词优先: 如果一个词同时在谓语族和其他族, 谓语最高置信度
        is_verb = word in self.verb_words or word in self.essential_verbs
        if word in self.subj_words:
            results.append(("主语", 0.95 if not is_verb else 0.80))
        if is_verb:
            results.append(("谓语", 0.95))  # 谓语族最高优先级
        if word in self.obj_words:
            results.append(("宾语", 0.95 if not is_verb else 0.75))
        if word in self.adj_words:
            results.append(("定语", 0.85 if not is_verb else 0.60))
        if word in self.adv_words:
            results.append(("状语", 0.85 if not is_verb else 0.60))
        if word in self.comp_words:
            results.append(("补语", 0.80))
        if word in self.func_words:
            results.append(("虚词", 0.85))

        # 规则2: 代词→主语候选优先
        if word in self.pronouns:
            if ("主语", 0.90) not in results:
                results.append(("主语", 0.85))
            # 代词也可能是宾语
            if word in {"我", "你", "他", "她"}:
                results.append(("宾语", 0.30))

        # 规则3: 位置启发式 (如果有context中的位置信息)
        if context:
            pos_in_sentence = context.get("position", -1)
            sentence_len = context.get("sentence_length", 0)

            if sentence_len == 3:
                if pos_in_sentence == 0:
                    results.append(("主语", 0.60))
                elif pos_in_sentence == 1:
                    results.append(("谓语", 0.60))
                elif pos_in_sentence == 2:
                    results.append(("宾语", 0.60))

        # 规则4: 没有匹配时默认
        if not results:
            results.append(("宾语", 0.20))
            results.append(("主语", 0.10))

        # 去重: 同一角色保留最高置信度
        merged = {}
        for role, conf in results:
            if role not in merged or conf > merged[role]:
                merged[role] = conf

        return {"syntax_candidates": sorted(merged.items(), key=lambda x: -x[1])}
