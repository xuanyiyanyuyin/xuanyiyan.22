"""
P3-C: 连接/逻辑关系 — P3第11子模块
====================================
连词、逻辑关系词、关联词对检测。
纯规则引擎, 0参数。
"""

class P3C_ConnectLogic:
    # 因果关系
    CAUSE = {"因为", "由于", "因", "所以", "因此", "因而", "以至于", "以致", "从而", "故", "之所以"}
    # 转折关系
    ADVERSATIVE = {"但是", "但", "可是", "然而", "却", "不过", "只是", "而", "然", "偏偏", "反倒", "反而", "倒是"}
    # 并列关系
    COORDINATE = {"和", "与", "及", "以及", "并且", "而且", "且", "而", "又", "也", "同时", "同样", "此外", "另外", "还有"}
    # 条件关系
    CONDITIONAL = {"如果", "假如", "倘若", "若", "要是", "只要", "只有", "除非", "无论", "不管", "不论", "即使", "即便", "哪怕", "就算"}
    # 递进关系
    PROGRESSIVE = {"不但", "不仅", "不只", "不光", "而且", "并且", "甚至", "甚而", "乃至", "况且", "何况", "再说", "再者"}
    # 让步关系
    CONCESSIVE = {"虽然", "虽说", "尽管", "固然", "即使", "即便", "哪怕", "就是", "纵使", "纵然"}
    # 选择关系
    ALTERNATIVE = {"或者", "或", "要么", "还是", "不是", "就是", "宁可", "宁愿", "与其", "不如", "毋宁"}
    # 承接/顺承
    SEQUENTIAL = {"然后", "接着", "于是", "便", "就", "才", "随后", "随即", "之后", "其后", "此后", "以后", "之后", "接下来", "进而"}
    # 总结
    SUMMARY = {"总之", "总而言之", "综上所述", "综上", "简言之", "一句话", "一言蔽之", "概括", "总归", "归根到底", "说到底"}
    # 举例
    EXAMPLE = {"例如", "比如", "譬如", "举例", "比方", "如", "像", "诸如", "以", "为例"}
    # 目的
    PURPOSE = {"为了", "为", "以便", "以免", "免得", "省得", "以求", "借以"}

    PAIRS = {
        ("因为", "所以"): "cause_effect",
        ("由于", "因此"): "cause_effect",
        ("虽然", "但是"): "concessive_adversative",
        ("尽管", "但是"): "concessive_adversative",
        ("如果", "就"): "condition_result",
        ("只要", "就"): "condition_result",
        ("只有", "才"): "condition_result",
        ("即使", "也"): "concessive",
        ("无论", "都"): "unconditional",
        ("不但", "而且"): "progressive",
        ("不仅", "而且"): "progressive",
        ("与其", "不如"): "preference",
        ("宁可", "也不"): "preference",
        ("为了", "以便"): "purpose",
        ("之所以", "是因为"): "explanation",
    }

    CATEGORIES = {
        "cause": CAUSE, "adversative": ADVERSATIVE, "coordinate": COORDINATE,
        "conditional": CONDITIONAL, "progressive": PROGRESSIVE, "concessive": CONCESSIVE,
        "alternative": ALTERNATIVE, "sequential": SEQUENTIAL, "summary": SUMMARY,
        "example": EXAMPLE, "purpose": PURPOSE,
    }

    def process(self, word, context=None):
        r = {"is_connective": False, "conn_type": "", "conn_confidence": 0.0}

        for cat, words_set in self.CATEGORIES.items():
            if word in words_set:
                r["is_connective"] = True
                r["conn_type"] = cat
                r["conn_confidence"] = 0.85
                return r
        return r

    def check_pair(self, words):
        """检测关联词对"""
        found = []
        for i, w in enumerate(words):
            res = self.process(w)
            if res["is_connective"]:
                found.append((i, w, res["conn_type"]))
        pairs = []
        for i in range(len(found)):
            for j in range(i+1, len(found)):
                w1, w2 = found[i][1], found[j][1]
                if (w1, w2) in self.PAIRS:
                    pairs.append((w1, w2, self.PAIRS[(w1, w2)]))
        return pairs


if __name__ == "__main__":
    p3c = P3C_ConnectLogic()
    tests = [
        ["因为", "loss", "不", "下降", "所以", "调", "学习率"],
        ["虽然", "准确率", "低", "但是", "在", "上升"],
        ["如果", "显存", "不够", "就", "降", "batch"],
        ["不仅", "省", "显存", "而且", "速度", "快"],
        ["首先", "加载", "数据", "然后", "初始化", "模型"],
    ]
    for words in tests:
        conns = [(w, p3c.process(w)["conn_type"]) for w in words if p3c.process(w)["is_connective"]]
        pairs = p3c.check_pair(words)
        print(f"'{''.join(words)}' → 连接词={conns} 配对={pairs}")
