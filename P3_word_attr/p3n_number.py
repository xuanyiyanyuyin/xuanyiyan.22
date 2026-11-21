"""
P3-N: 数量/程度属性 — P3第10子模块
====================================
数词、量词、程度副词、约数表达。
纯规则引擎, 0参数。
"""

class P3N_NumberDegree:
    # 数词
    NUMERALS = {'零','一','二','三','四','五','六','七','八','九','十','百','千','万','亿','兆',
                '两','廿','卅','壹','贰','叁','肆','伍','陆','柒','捌','玖','拾','佰','仟'}
    DIGITS = set('0123456789０１２３４５６７８９')
    # 量词
    MEASURE = {'个','只','条','张','把','台','件','块','篇','本','首','幅','座','栋','棵',
               '粒','滴','片','朵','阵','场','次','遍','回','趟','下','顿','番','通',
               '年','月','日','天','时','分','秒','小时','分钟','秒钟',
               '米','厘米','毫米','千米','公里','斤','公斤','克','吨','升','毫升',
               '元','角','分','块','毛','度','瓦','赫','帕','比特','字节',
               '位','名','种','类','样','套','组','批','群','堆','双','对','打',
               '口','头','匹','架','艘','辆','册','期','届','任','等','级','层','段',}
    # 序数词
    ORDINAL = {'第','初','头','末','首','次','前','后'}
    # 程度副词
    DEGREE_HIGH = {'很','非常','十分','特别','极其','格外','异常','相当','挺','蛮','太','极','极度',
                   '最为','最为','最为','高度','深入','深刻','严重','重大','巨大','极大','极强'}
    DEGREE_MED = {'比较','较','有些','有点','稍微','略微','多少','不太','不怎么','还算','还算','还'}
    DEGREE_LOW = {'略微','稍','稍微','稍稍','略','些许','些微','微微','有点儿'}
    DEGREE_EXCESS = {'过','过于','过分','太','忒','超'}
    # 约数
    APPROX = {'大约','大概','左右','上下','将近','差不多','近乎','约','许','余','几','多','来','左右',
              '若干','一些','一些','大量','少量','部分','多数','少数','绝大多数','极少数',
              '所有','全部','整个','全','整','一切','凡是','任何','每','各','所有','全体',
              '一半','三分之一','百分之','万分之','折','成','倍',
              '第一','第二','第三','其次','再次','最后','最初',
              '只','仅仅','仅','不过','只有','光','单','不止','超过','不止','不止'}

    def process(self, word, context=None):
        r = {"is_number": False, "num_type": "", "num_confidence": 0.0,
             "is_measure": False, "is_degree": False, "degree_level": "",
             "is_quantifier": False}

        if word in self.NUMERALS:
            r["is_number"] = True; r["num_type"] = "chinese_numeral"; r["num_confidence"] = 0.9
        elif all(c in self.DIGITS for c in word):
            r["is_number"] = True; r["num_type"] = "digit"; r["num_confidence"] = 0.95

        if word in self.MEASURE:
            r["is_measure"] = True; r["num_confidence"] = max(r["num_confidence"], 0.85)

        if word in self.ORDINAL:
            r["is_number"] = True; r["num_type"] = "ordinal"; r["num_confidence"] = 0.8

        if word in self.DEGREE_HIGH:
            r["is_degree"] = True; r["degree_level"] = "high"; r["num_confidence"] = max(r["num_confidence"], 0.9)
        elif word in self.DEGREE_MED:
            r["is_degree"] = True; r["degree_level"] = "medium"; r["num_confidence"] = max(r["num_confidence"], 0.8)
        elif word in self.DEGREE_LOW:
            r["is_degree"] = True; r["degree_level"] = "low"; r["num_confidence"] = max(r["num_confidence"], 0.7)
        elif word in self.DEGREE_EXCESS:
            r["is_degree"] = True; r["degree_level"] = "excess"; r["num_confidence"] = max(r["num_confidence"], 0.85)

        if word in self.APPROX:
            r["is_quantifier"] = True; r["num_confidence"] = max(r["num_confidence"], 0.8)

        return r


if __name__ == "__main__":
    p3n = P3N_NumberDegree()
    tests = [
        ["损失", "只有", "0", ".", "18", "%"],
        ["训练", "了", "三千", "轮"],
        ["这个", "非常", "重要"],
        ["大约", "五十", "个", "样本"],
        ["第一", "次", "跑", "完全", "收敛"],
    ]
    for words in tests:
        res = p3n.process("_".join(words))
        hits = [(w, r) for w, r in zip(words, [p3n.process(w) for w in words]) if any(r.values())]
        tags = []
        for w, r in hits:
            if r["is_number"]: tags.append(f"{w}=num")
            if r["is_measure"]: tags.append(f"{w}=meas")
            if r["is_degree"]: tags.append(f"{w}=deg({r['degree_level']})")
            if r["is_quantifier"]: tags.append(f"{w}=quant")
        print(f"'{''.join(words)}' → {tags}")
