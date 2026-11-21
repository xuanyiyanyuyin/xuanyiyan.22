"""
P3-P: 标点符号属性 — P3第14子模块
====================================
中文标点承载句法边界、语气、停顿层级。
纯规则引擎, 0参数。
"""

class P3P_Punctuation:
    # 句末标点 → 句类型直接信号
    SENT_END = {
        "。": ("period", "statement", 0.95),
        "！": ("exclamation", "exclamation", 0.95),
        "？": ("question", "question", 0.95),
        "?": ("question", "question", 0.9),
        "!": ("exclamation", "exclamation", 0.9),
        "…": ("ellipsis", "unfinished", 0.8),
        "......": ("ellipsis", "unfinished", 0.8),
        "~": ("tilde", "playful", 0.6),
    }
    # 句中分隔 → 层级/逻辑
    SENT_MID = {
        "，": ("comma", "clause_boundary", 0.9),
        ",": ("comma", "clause_boundary", 0.8),
        "；": ("semicolon", "parallel_clause", 0.9),
        ";": ("semicolon", "parallel_clause", 0.8),
        "：": ("colon", "explanation", 0.85),
        ":": ("colon", "explanation", 0.8),
        "、": ("enum_comma", "enumeration", 0.9),
        "．": ("dot", "abbreviation", 0.6),
    }
    # 配对/引号
    PAIRED = {
        "「": ("quote_open", "dialogue", "corner"),
        "」": ("quote_close", "dialogue", "corner"),
        "『": ("quote_open", "title", "double_corner"),
        "』": ("quote_close", "title", "double_corner"),
        """: ("quote_open", "quotation", "double"),
        """: ("quote_close", "quotation", "double"),
        "'": ("quote_open", "quotation", "single"),
        "'": ("quote_close", "quotation", "single"),
        "（": ("paren_open", "parenthesis", "fullwidth"),
        "）": ("paren_close", "parenthesis", "fullwidth"),
        "(": ("paren_open", "parenthesis", "halfwidth"),
        ")": ("paren_close", "parenthesis", "halfwidth"),
        "《": ("bracket_open", "title", "book"),
        "》": ("bracket_close", "title", "book"),
        "【": ("bracket_open", "emphasis", "square"),
        "】": ("bracket_close", "emphasis", "square"),
        "〔": ("bracket_open", "annotation", "tortoise"),
        "〕": ("bracket_close", "annotation", "tortoise"),
    }
    # 连接/破折
    DASH = {
        "—": ("em_dash", "break", 0.9),
        "–": ("en_dash", "range", 0.8),
        "——": ("em_dash", "break", 0.95),
        "-": ("hyphen", "compound", 0.6),
        "·": ("middle_dot", "name_separator", 0.8),
    }
    # 特殊符号
    SPECIAL = {
        "%": ("percent", "ratio", 0.9),
        "％": ("percent", "ratio", 0.95),
        "+": ("plus", "operation", 0.7),
        "-": ("minus", "operation", 0.6),
        "=": ("equals", "definition", 0.7),
        "/": ("slash", "alternative", 0.6),
        "／": ("slash", "alternative", 0.7),
        "@": ("at", "mention", 0.8),
        "#": ("hash", "tag", 0.7),
        "＃": ("hash", "tag", 0.8),
        "&": ("ampersand", "conjunction", 0.7),
        "*": ("asterisk", "footnote", 0.6),
        "＊": ("asterisk", "footnote", 0.7),
        "_": ("underscore", "emphasis", 0.5),
        "＿": ("underscore", "emphasis", 0.6),
        "^": ("caret", "exponent", 0.6),
        "＾": ("caret", "exponent", 0.7),
        "$": ("dollar", "currency", 0.8),
        "￥": ("yen", "currency", 0.9),
        "€": ("euro", "currency", 0.8),
        "°": ("degree", "unit", 0.8),
        "℃": ("celsius", "unit", 0.9),
        "′": ("prime", "unit", 0.7),
        "″": ("double_prime", "unit", 0.7),
        "×": ("times", "operation", 0.8),
        "÷": ("divide", "operation", 0.8),
        "±": ("plus_minus", "tolerance", 0.8),
        "∞": ("infinity", "concept", 0.7),
    }
    # 中文特有标点
    CN_PUNCT = {
        "。", "！", "？", "，", "；", "：", "、", "…", "——",
        "「", "」", "『", "』", """, """, "'", "'",
        "（", "）", "《", "》", "【", "】", "〔", "〕",
        "·", "％", "＋", "－", "＝", "／", "＃", "＊", "＿", "＾",
    }

    def process(self, word, context=None):
        r = {"is_punct": False, "punct_type": "", "punct_role": "",
             "punct_subtype": "", "confidence": 0.0, "is_sent_boundary": False}

        if word in self.SENT_END:
            ptype, role, conf = self.SENT_END[word]
            r["is_punct"] = True; r["punct_type"] = ptype; r["punct_role"] = role
            r["confidence"] = conf; r["is_sent_boundary"] = True

        elif word in self.SENT_MID:
            ptype, role, conf = self.SENT_MID[word]
            r["is_punct"] = True; r["punct_type"] = ptype; r["punct_role"] = role
            r["confidence"] = conf

        elif word in self.PAIRED:
            ptype, role, subtype = self.PAIRED[word]
            r["is_punct"] = True; r["punct_type"] = ptype; r["punct_role"] = role
            r["punct_subtype"] = subtype; r["confidence"] = 0.85

        elif word in self.DASH:
            ptype, role, conf = self.DASH[word]
            r["is_punct"] = True; r["punct_type"] = ptype; r["punct_role"] = role
            r["confidence"] = conf

        elif word in self.SPECIAL:
            ptype, role, conf = self.SPECIAL[word]
            r["is_punct"] = True; r["punct_type"] = ptype; r["punct_role"] = role
            r["confidence"] = conf

        elif word in self.CN_PUNCT:
            r["is_punct"] = True; r["punct_type"] = "other_cn"; r["confidence"] = 0.7

        return r

    def sent_boundary_positions(self, words):
        """返回句边界位置"""
        return [i for i, w in enumerate(words) if self.process(w)["is_sent_boundary"]]

    def punct_ratio(self, words):
        """标点占比"""
        if not words: return 0
        return sum(1 for w in words if self.process(w)["is_punct"]) / len(words)


if __name__ == "__main__":
    p3p = P3P_Punctuation()
    tests = [
        ["今天", "天气", "真好", "！"],
        ["你", "喜欢", "什么", "电影", "？"],
        ["模型", "收敛", "了", "，", "但是", "loss", "还", "很高", "。"],
        ["GPU", "利用率", "只有", "18", "%", "。"],
        ["P3", "-", "T", "：", "时态", "检测"],
    ]
    for words in tests:
        tags = [(w, p3p.process(w)["punct_role"]) for w in words if p3p.process(w)["is_punct"]]
        boundaries = p3p.sent_boundary_positions(words)
        print(f"'{''.join(words)}' → 标点={tags} 句边界={boundaries}")
