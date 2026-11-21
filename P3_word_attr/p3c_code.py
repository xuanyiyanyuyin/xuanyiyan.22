"""
P3-C: 代码属性检测 — P3第8子模块
===================================
识别词是否属于代码/技术语言, 并标注代码属性。
纯规则引擎, 0参数, 与P3其他模块接口一致。
"""
import re
from typing import Dict


class P3C_CodeDetect:
    """代码/技术语言属性检测器"""

    # 编程语言关键词
    PY_KEYWORDS = {
        "def", "class", "return", "if", "else", "elif", "for", "while",
        "import", "from", "as", "try", "except", "finally", "with",
        "yield", "lambda", "pass", "break", "continue", "raise",
        "and", "or", "not", "in", "is", "None", "True", "False",
        "self", "super", "global", "nonlocal", "async", "await",
        "print", "len", "range", "int", "str", "float", "list", "dict",
        "set", "tuple", "bool", "type", "open", "input", "enumerate",
        "zip", "map", "filter", "sorted", "reversed", "any", "all",
    }
    PY_BUILTINS = {
        "int", "str", "float", "list", "dict", "set", "tuple", "bool",
        "print", "len", "range", "type", "open", "input", "enumerate",
        "zip", "map", "filter", "sorted", "reversed", "any", "all",
        "abs", "round", "min", "max", "sum", "pow", "divmod",
    }
    JS_KEYWORDS = {
        "function", "var", "let", "const", "return", "if", "else",
        "for", "while", "do", "switch", "case", "break", "continue",
        "new", "this", "typeof", "instanceof", "try", "catch",
        "finally", "throw", "async", "await", "export", "import",
        "class", "extends", "super", "static", "get", "set",
        "null", "undefined", "true", "false", "console",
    }
    CPP_KEYWORDS = {
        "int", "float", "double", "char", "void", "bool", "long",
        "short", "unsigned", "signed", "struct", "class", "enum",
        "union", "typedef", "sizeof", "new", "delete", "this",
        "virtual", "override", "final", "const", "static", "extern",
        "if", "else", "for", "while", "do", "switch", "case",
        "break", "continue", "return", "goto", "namespace", "using",
        "template", "typename", "public", "private", "protected",
        "include", "define", "ifdef", "ifndef", "endif",
    }
    SQL_KEYWORDS = {
        "select", "from", "where", "insert", "update", "delete",
        "create", "drop", "alter", "table", "index", "view",
        "join", "left", "right", "inner", "outer", "on", "group",
        "by", "order", "having", "limit", "offset", "union",
        "all", "distinct", "as", "into", "values", "set",
        "and", "or", "not", "null", "is", "like", "between", "in",
        "count", "sum", "avg", "max", "min", "exists",
    }
    # HTML/CSS
    HTML_TAGS = {
        "div", "span", "p", "a", "img", "ul", "ol", "li", "table",
        "tr", "td", "th", "form", "input", "button", "label",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "footer",
        "nav", "section", "article", "main", "body", "head", "html",
        "script", "style", "link", "meta", "title",
    }

    # 操作符模式
    OP_PATTERNS = [
        (r'^[+\-*/%&|^~<>!=]=?$', "operator"),
        (r'^[<>]=?$', "comparison"),
        (r'^[&|]{2}$', "logical_op"),
        (r'^[+\-]{2}$', "increment"),
        (r'^[(){}\[\];,.:]$', "delimiter"),
    ]
    # 字面量模式
    LITERAL_PATTERNS = [
        (r'^\d+$', "number_literal"),
        (r'^\d+\.\d+$', "float_literal"),
        (r'^["\'].*["\']$', "string_literal"),
        (r'^0x[0-9a-fA-F]+$', "hex_literal"),
        (r'^(True|False|None|null|undefined)$', "bool_literal"),
    ]
    # 标识符模式
    ID_PATTERNS = [
        (r'^[a-zA-Z_][a-zA-Z0-9_]*$', "identifier"),
        (r'^[a-z][a-zA-Z0-9]*$', "camel_case"),
        (r'^[a-z][a-zA-Z0-9]*_+[a-z]', "snake_case"),
        (r'^[A-Z][a-zA-Z0-9]*$', "pascal_case"),
    ]
    # 中文编程/技术词汇
    CN_TECH = {
        "代码", "函数", "变量", "参数", "返回", "循环", "条件",
        "数组", "列表", "字典", "字符串", "整数", "浮点", "布尔",
        "对象", "类", "实例", "方法", "属性", "模块", "包",
        "导入", "异常", "错误", "调试", "测试", "编译", "运行",
        "接口", "继承", "重载", "多态", "封装", "抽象",
        "算法", "数据结构", "排序", "搜索", "遍历", "递归",
        "数据库", "查询", "索引", "缓存", "线程", "进程",
        "网络", "请求", "响应", "接口", "协议", "序列化",
        "前端", "后端", "框架", "组件", "渲染", "路由",
        "模型", "训练", "预测", "损失", "梯度", "优化",
        "嵌入", "向量", "矩阵", "维度", "张量", "归一化",
        "激活", "卷积", "池化", "注意力", "编码器", "解码器",
        "参数", "权重", "偏置", "学习率", "批次", "轮数",
    }

    def process(self, word: str, context: dict = None) -> dict:
        """处理单个词, 返回代码属性字典"""
        result = {
            "is_code": False,
            "code_confidence": 0.0,
            "code_type": "",           # keyword/identifier/operator/literal/comment/tech_term
            "language": "",            # python/javascript/cpp/sql/html/general
            "code_role": "",           # declaration/control_flow/function_call/data_access/other
            "cn_tech": False,          # 中文技术词汇
        }

        # 1. 关键词检测
        for kw_set, lang_name in [
            (self.PY_KEYWORDS, "python"),
            (self.JS_KEYWORDS, "javascript"),
            (self.CPP_KEYWORDS, "cpp"),
            (self.SQL_KEYWORDS, "sql"),
            (self.HTML_TAGS, "html"),
        ]:
            if word.lower() in kw_set:
                result["is_code"] = True
                result["code_confidence"] = 0.95
                result["code_type"] = "keyword"
                result["language"] = lang_name
                result["code_role"] = self._keyword_role(word.lower())
                return result

        # 2. 操作符检测
        for pattern, ctype in self.OP_PATTERNS:
            if re.match(pattern, word):
                result["is_code"] = True
                result["code_confidence"] = 0.9
                result["code_type"] = ctype
                result["language"] = "general"
                result["code_role"] = "operation"
                return result

        # 3. 字面量检测
        for pattern, ctype in self.LITERAL_PATTERNS:
            if re.match(pattern, word):
                result["is_code"] = True
                result["code_confidence"] = 0.9
                result["code_type"] = ctype
                result["language"] = "general"
                result["code_role"] = "data"
                return result

        # 4. 标识符检测
        for pattern, ctype in self.ID_PATTERNS:
            if re.match(pattern, word) and len(word) >= 2:
                result["is_code"] = True
                result["code_confidence"] = 0.7
                result["code_type"] = ctype
                result["language"] = "general"
                result["code_role"] = "data_reference"
                return result

        # 5. 中文技术词汇
        if word in self.CN_TECH:
            result["is_code"] = True
            result["code_confidence"] = 0.85
            result["code_type"] = "tech_term"
            result["language"] = "chinese_tech"
            result["code_role"] = "concept"
            result["cn_tech"] = True

        return result

    def _keyword_role(self, kw: str) -> str:
        """推断关键词的代码角色"""
        declaration = {"def", "class", "function", "var", "let", "const", "int",
                       "float", "double", "char", "void", "struct", "enum", "typedef"}
        control_flow = {"if", "else", "elif", "for", "while", "do", "switch",
                        "case", "break", "continue", "return", "goto", "yield",
                        "try", "except", "catch", "finally", "throw", "raise",
                        "async", "await"}
        function_call = {"print", "len", "range", "open", "input", "enumerate",
                         "zip", "map", "filter", "sorted", "reversed",
                         "count", "sum", "avg", "max", "min", "select", "insert",
                         "update", "delete", "create", "drop"}
        import_export = {"import", "from", "export", "include", "require", "using", "namespace"}
        data_access = {"self", "this", "super", "typeof", "instanceof", "sizeof", "new", "delete"}

        if kw in declaration: return "declaration"
        if kw in control_flow: return "control_flow"
        if kw in function_call: return "function_call"
        if kw in import_export: return "import_export"
        if kw in data_access: return "data_access"
        return "other"

    def process_sentence(self, words: list) -> list:
        """处理整句, 返回每个词的代码属性"""
        return [self.process(w) for w in words]

    def code_ratio(self, words: list) -> float:
        """返回句子中代码词的比例"""
        results = self.process_sentence(words)
        if not results:
            return 0.0
        return sum(1 for r in results if r["is_code"]) / len(results)

    def detect_language(self, words: list) -> str:
        """检测代码语言(句子级)"""
        results = self.process_sentence(words)
        langs = {}
        for r in results:
            l = r.get("language", "")
            if l and l != "general":
                langs[l] = langs.get(l, 0) + 1
        if langs:
            return max(langs, key=langs.get)
        return "mixed" if self.code_ratio(words) > 0.3 else "natural"


# ── 测试 ──
if __name__ == "__main__":
    p3c = P3C_CodeDetect()

    tests = [
        # 中文+代码混合
        ["def", "train", "(", "model", ",", "data", ")", ":"],
        ["print", "(", "'hello'", ")"],
        ["我", "用", "pytorch", "训练", "了", "模型"],
        ["class", "Transformer", "(", "nn", ".", "Module", ")", ":"],
        ["select", "*", "from", "users", "where", "id", "=", "1"],
        ["如果", "loss", "不", "下降", "就", "调", "学习率"],
        ["GPU", "利用率", "只有", "18", "%"],
        ["import", "torch", ";", "import", "numpy"],
    ]

    for i, words in enumerate(tests):
        results = p3c.process_sentence(words)
        ratio = p3c.code_ratio(words)
        lang = p3c.detect_language(words)
        code_words = [w for w, r in zip(words, results) if r["is_code"]]
        roles = [(w, r["code_role"]) for w, r in zip(words, results) if r["is_code"]]
        print(f"\n[{i+1}] '{''.join(words)}'")
        print(f"   代码词={code_words} 代码比={ratio:.0%} 语言={lang}")
        print(f"   角色: {roles}")
