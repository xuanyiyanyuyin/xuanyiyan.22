"""
P3-A: 基础词类属性
判断词的基础类型: 名词/动词/代词/疑问词/地点词/时间词/方式词/虚词
"""
from typing import Dict, Tuple


class P3A_BasicType:
    """基础词类判定 — 基于词表查詢 + 启发式规则"""

    def __init__(self):
        # 词类词表
        self.pronouns = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
                         "自己", "大家", "别人", "谁", "什么", "哪个", "这位", "那位", "其"}
        self.question_words = {"谁", "什么", "哪里", "怎么", "为什么", "什么时候",
                              "哪个", "哪些", "怎样", "多少", "多久", "多远"}
        self.place_words = {"学校", "医院", "商店", "餐厅", "公园", "超市", "银行", "图书馆",
                           "房间", "工厂", "城市", "房屋", "桥梁", "道路", "这里", "那里",
                           "家里", "办公室", "教室", "操场", "车站", "机场",
                           "山","水","海","河","湖","岛","村","城","街","路","门","窗","楼","室","厅","堂","院","园","地","田"}
        self.time_words = {"今天", "明天", "昨天", "现在", "以前", "以后", "早上", "晚上",
                          "上午", "下午", "时候", "时间", "小时", "分钟", "星期", "月", "年"}
        self.manner_words = {"怎么", "怎样", "如此", "如何", "这么", "那么", "这样", "那样",
                            "慢慢", "快快", "轻轻", "重重"}
        self.function_words = {"的", "了", "在", "是", "有", "和", "与", "或", "不", "也",
                              "都", "就", "才", "还", "又", "再", "吗", "呢", "吧", "啊",
                              "着", "过", "被", "把", "让", "给", "对", "从", "到", "向"}

    def process(self, word: str, context: dict = None) -> Dict:
        """返回: {basic_type, confidence}"""

        if word in self.question_words:
            # 细分类: 人物疑问/事物疑问/地点疑问/方式疑问
            if word in {"谁", "哪个", "哪些"}:
                return {"basic_type": "question_person", "confidence": 0.99}
            elif word in {"哪里", "哪儿"}:
                return {"basic_type": "question_place", "confidence": 0.99}
            elif word in {"怎么", "怎样", "如何"}:
                return {"basic_type": "question_manner", "confidence": 0.99}
            elif word in {"为什么"}:
                return {"basic_type": "question_reason", "confidence": 0.99}
            elif word in {"什么时候", "多久"}:
                return {"basic_type": "question_time", "confidence": 0.99}
            else:
                return {"basic_type": "question_thing", "confidence": 0.99}

        if word in self.pronouns:
            return {"basic_type": "pronoun", "confidence": 0.99}

        if word in self.function_words:
            return {"basic_type": "function", "confidence": 0.95}

        if word in self.place_words:
            return {"basic_type": "noun_place", "confidence": 0.90}

        if word in self.time_words:
            return {"basic_type": "noun_time", "confidence": 0.90}

        if word in self.manner_words:
            return {"basic_type": "manner", "confidence": 0.85}

        # 启发式: 常见单字直接分类
        common_nouns = {"人","手","脚","头","眼","耳","鼻","口","舌","牙","身","心","血","骨","皮","毛","发","衣","鞋","帽","饭","菜","茶","酒","书","笔","纸","灯","车","船","狗","猫","牛","马","羊","鱼","鸟","虫","草","木","石","金","银","铜","铁","风","雨","云","雪","雷","电","星","月","日","天","地","山","水","火","土","花","叶","根","果","米","面","肉","蛋","奶","油","盐","糖","醋","酒","药","针","线","布","丝","网","绳","袋","盒","瓶","罐","桶","盆","碗","盘","杯","壶","锅","炉","床","桌","椅","柜","架","箱","包","袋"}
        common_verbs = {"吃","喝","看","写","去","买","做","玩","听","说","走","跑","跳","唱","画","读","想","打","拿","放","开","关","进","出","上","下","来","回","过","到","停","坐","站","睡","醒","飞","游","送","收","交","还","借","租","卖","付","花","省","赚","赔","赢","输","胜","败","教","学","问","答","帮","救","杀","死","给","让","等","待","找","查","算","数","计","量","测","试","验","修","改","装","拆","洗","擦","扫","拖","搬","抬","推","拉","挂","贴","插","拔","切","割","磨","刨","钻","焊","铸","锻","压","轧"}
        common_adjs = {"好","坏","大","小","高","低","长","短","快","慢","多","少","远","近","深","浅","轻","重","厚","薄","宽","窄","硬","软","干","湿","冷","热","凉","暖","新","旧","老","少","美","丑","胖","瘦","富","穷","贵","贱","强","弱","对","错","真","假","正","反","直","弯","平","斜","方","圆","粗","细","松","紧","稀","密","浓","淡","清","浊","明","暗","亮","黑"}
        if word in common_nouns:
            return {"basic_type": "noun", "confidence": 0.75}
        if word in common_verbs:
            return {"basic_type": "verb", "confidence": 0.75}
        if word in common_adjs:
            return {"basic_type": "adjective", "confidence": 0.70}
        # 结尾字判断
        if word.endswith("的"):
            return {"basic_type": "adjective", "confidence": 0.60}
        if word.endswith("了") or word.endswith("着") or word.endswith("过"):
            return {"basic_type": "verb", "confidence": 0.50}

        # 默认: 可能是名词或动词
        return {"basic_type": "content_word", "confidence": 0.30}
