"""
P3-V: 语态/语气 — P3第12子模块
================================
被动/把字/语气词/情态动词。
纯规则引擎, 0参数。
"""

class P3V_VoiceMood:
    # 被动标记
    PASSIVE_MARKERS = {"被", "叫", "让", "给", "受到", "遭到", "得到", "遭受", "蒙受"}
    # 把字/处置式
    DISPOSAL = {"把", "将"}
    # 语气助词 (句末)
    MOOD_PARTICLES = {
        "吧": ("suggestion", 0.9),
        "呢": ("inquiry", 0.7),
        "吗": ("yes_no_question", 0.95),
        "啊": ("exclamation", 0.7),
        "呀": ("exclamation", 0.75),
        "嘛": ("obvious", 0.8),
        "哦": ("reminder", 0.7),
        "呗": ("casual", 0.8),
        "咯": ("casual", 0.75),
        "罢了": ("dismissive", 0.85),
        "而已": ("downplay", 0.9),
        "来着": ("reminiscence", 0.85),
        "的话": ("conditional", 0.8),
        "得了": ("permission", 0.75),
        "算了": ("give_up", 0.85),
        "不成": ("rhetorical", 0.8),
        "不是": ("tag_question", 0.7),
    }
    # 情态动词
    MODAL_VERBS = {
        "能": ("ability", 0.85),
        "能够": ("ability", 0.9),
        "可以": ("permission", 0.85),
        "会": ("capability", 0.8),
        "可能": ("possibility", 0.85),
        "应该": ("obligation", 0.9),
        "应当": ("obligation", 0.85),
        "必须": ("necessity", 0.95),
        "得": ("necessity", 0.7),
        "需要": ("need", 0.8),
        "要": ("volition", 0.7),
        "想": ("desire", 0.75),
        "想要": ("desire", 0.85),
        "希望": ("hope", 0.85),
        "愿意": ("willingness", 0.9),
        "肯": ("willingness", 0.8),
        "敢": ("dare", 0.85),
        "也许": ("possibility", 0.8),
        "或许": ("possibility", 0.8),
        "大概": ("probability", 0.75),
        "似乎": ("appearance", 0.8),
        "好像": ("appearance", 0.75),
        "仿佛": ("appearance", 0.8),
    }

    def process(self, word, context=None):
        r = {"voice_type": "", "mood_type": "", "modal_type": "", "confidence": 0.0}

        if word in self.PASSIVE_MARKERS:
            r["voice_type"] = "passive"; r["confidence"] = 0.85

        if word in self.DISPOSAL:
            r["voice_type"] = "disposal"; r["confidence"] = 0.9

        if word in self.MOOD_PARTICLES:
            mood, conf = self.MOOD_PARTICLES[word]
            r["mood_type"] = mood; r["confidence"] = max(r["confidence"], conf)

        if word in self.MODAL_VERBS:
            modal, conf = self.MODAL_VERBS[word]
            r["modal_type"] = modal; r["confidence"] = max(r["confidence"], conf)

        return r


if __name__ == "__main__":
    p3v = P3V_VoiceMood()
    tests = [
        ["模型", "被", "训练", "了", "5000", "轮"],
        ["把", "batch", "调", "大", "吧"],
        ["显存", "可以", "再", "加", "吗"],
        ["我们", "应该", "能", "跑完"],
        ["这个", "似乎", "收敛", "了", "呢"],
    ]
    for words in tests:
        hits = [(w, p3v.process(w)) for w in words]
        tags = []
        for w, r in hits:
            if r["voice_type"]: tags.append(f"{w}=voice({r['voice_type']})")
            if r["mood_type"]: tags.append(f"{w}=mood({r['mood_type']})")
            if r["modal_type"]: tags.append(f"{w}=modal({r['modal_type']})")
        print(f"'{''.join(words)}' → {tags}")
