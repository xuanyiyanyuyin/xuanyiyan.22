"""
P3-M: 修饰/限定 — P3第13子模块
================================
形容词、副词、定语/状语标记(的/地/得)、比较级。
纯规则引擎, 0参数。
"""

class P3M_Modifier:
    # 的/地/得 — 中文核心修饰标记
    DE = {"的": "attributive", "地": "adverbial", "得": "complement"}
    # 形容词
    ADJ = {
        "好", "坏", "大", "小", "多", "少", "高", "低", "长", "短",
        "快", "慢", "新", "旧", "老", "年轻", "远", "近", "深", "浅",
        "强", "弱", "难", "易", "简单", "复杂", "容易", "困难",
        "重要", "主要", "次要", "关键", "核心", "基础",
        "正确", "错误", "对", "错", "真", "假",
        "清晰", "模糊", "明确", "含糊", "清楚", "明显",
        "稳定", "不稳", "可靠", "不可靠", "安全", "危险",
        "有效", "无效", "有用", "无用", "成功", "失败",
        "正常", "异常", "普通", "特殊", "一般", "特别",
        "全新", "强大", "弱小", "固定", "灵活", "严格", "宽松",
        "精确", "粗略", "细致", "粗糙", "紧密", "松散",
    }
    # 副词 (方式/状态)
    ADV_MANNER = {
        "快", "慢", "好", "坏", "轻", "重", "急", "缓", "稳", "准",
        "仔细", "认真", "努力", "积极", "主动", "被动",
        "直接", "间接", "逐步", "逐渐", "渐渐", "慢慢",
        "迅速", "快速", "急速", "缓慢", "飞快", "飞速",
        "自动", "手动", "手动", "半自动",
        "全面", "部分", "全部", "局部", "整体", "逐个",
        "反复", "重复", "不断", "不停", "持续", "连续",
        "单独", "共同", "一起", "一块", "分别", "各自",
    }
    # 比较级标记
    COMPARATIVE = {"更", "更加", "最为", "最", "比较", "较", "较为", "相对",
                   "越来越", "愈发", "日益", "尤其", "特别", "格外",
                   "不如", "不及", "比不上", "比", "比起", "相比",
                   "一样", "相同", "相似", "类似", "不同", "相反",
                   "差不多", "差不多", "近似", "接近", "等于"}
    # 范围副词
    SCOPE = {"都", "全", "全部", "所有", "整个", "基本上", "大部分",
             "大部分", "部分", "局部", "唯一", "仅仅", "只",
             "至少", "最多", "最少", "至多", "起码", "顶多",
             "总共", "一共", "合计", "累计", "总共",
             "平均", "均匀", "平衡", "均匀", "对称",
             "整整", "整整", "足足", "将近", "大约", "估值"}

    def process(self, word, context=None):
        r = {"is_modifier": False, "mod_type": "", "mod_confidence": 0.0,
             "is_de": False, "de_type": "", "is_comparative": False}

        if word in self.DE:
            r["is_de"] = True; r["is_modifier"] = True
            r["de_type"] = self.DE[word]; r["mod_confidence"] = 0.95
            return r

        if word in self.ADJ:
            r["is_modifier"] = True; r["mod_type"] = "adjective"; r["mod_confidence"] = 0.8
        elif word in self.ADV_MANNER:
            r["is_modifier"] = True; r["mod_type"] = "adverb_manner"; r["mod_confidence"] = 0.75
        elif word in self.SCOPE:
            r["is_modifier"] = True; r["mod_type"] = "scope"; r["mod_confidence"] = 0.8

        if word in self.COMPARATIVE:
            r["is_comparative"] = True
            r["mod_confidence"] = max(r["mod_confidence"], 0.8)

        return r


if __name__ == "__main__":
    p3m = P3M_Modifier()
    tests = [
        ["很", "好", "的", "结果"],
        ["快速", "地", "收敛"],
        ["跑", "得", "很", "快"],
        ["这个", "比", "那个", "更", "好"],
        ["全部", "数据", "都", "正确"],
    ]
    for words in tests:
        tags = []
        for w in words:
            r = p3m.process(w)
            if r["is_de"]: tags.append(f"{w}=de({r['de_type']})")
            if r["mod_type"]: tags.append(f"{w}={r['mod_type']}")
            if r["is_comparative"]: tags.append(f"{w}=comp")
        print(f"'{''.join(words)}' → {tags}")
