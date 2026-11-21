# -*- coding: utf-8 -*-
"""P3Code: 0参数规则式代码属性 (128D/字符) — 给V22 token的attr扩出代码模态。
槽位:
 0-7 字符类 one-hot: [字母,数字,下划线,运算符,括号,空白,'#/'(注释符),中文]
 8 行首缩进/8(clip1)  9 括号深度/8(clip1)  10 处于行内注释后  11 处于字符串内
 12 该行含中文  13 该行代码符号/字母占比  14-15 语言 one-hot [py, js]
 16-31 关键字 one-hot [def,class,return,if,else,for,while,import,from,print,self,None,True,False,function,const]
 32-127 预留0
"""
import torch

KW = ["def", "class", "return", "if", "else", "for", "while", "import", "from", "print", "self", "None", "True", "False", "function", "const"]
OPS = set("+-*/%=<>!&|^~")
BR = set("()[]{}")
CJK = lambda c: "\u4e00" <= c <= "\u9fff"


def _line_bounds(words, i):
    a = i
    while a > 0 and words[a - 1] != "\n":
        a -= 1
    b = i
    n = len(words)
    while b < n and words[b] != "\n":
        b += 1
    return a, b


def code_mat(words, L, device):
    vec = torch.zeros(L, 128)
    n = min(len(words), L)
    depth = 0
    for i in range(n):
        c = words[i]
        v = vec[i]
        la, lb = _line_bounds(words, i)
        line = "".join(words[la:lb])
        # char class
        if c.isalpha() and not CJK(c):
            v[0] = 1
        elif c.isdigit():
            v[1] = 1
        elif c == "_":
            v[2] = 1
        elif c in OPS:
            v[3] = 1
        elif c in BR:
            v[4] = 1
        elif c.isspace():
            v[5] = 1
        elif c in "#/":
            v[6] = 1
        elif CJK(c):
            v[7] = 1
        indent = len(line) - len(line.lstrip(" "))
        v[8] = min(indent / 8.0, 1.0)
        if c in BR:
            depth = max(0, depth + (1 if c in "([{" else -1))
        v[9] = min(depth / 8.0, 1.0)
        pre = "".join(words[la:i + 1])
        v[10] = 1.0 if ("#" in pre or "//" in pre) else 0.0
        v[11] = 1.0 if (pre.count('"') % 2 == 1 or pre.count("'") % 2 == 1) else 0.0
        v[12] = 1.0 if any(CJK(x) for x in line) else 0.0
        sym = sum(1 for x in line if x in OPS or x in BR)
        v[13] = min(sym / max(len(line), 1), 1.0)
        v[14] = 1.0 if ("def " in line or "import " in line or "self." in line) else 0.0
        v[15] = 1.0 if ("function " in line or "const " in line or "=>" in line) else 0.0
        # keyword at this position
        j = i
        while j > 0 and (words[j - 1].isalnum() or words[j - 1] == "_"):
            j -= 1
        k = i + 1
        while k < n and (words[k].isalnum() or words[k] == "_"):
            k += 1
        tok = "".join(words[j:k])
        if tok in KW:
            v[16 + KW.index(tok)] = 1.0
    return vec.to(device)

def code_np(item):
    words, L = item
    return code_mat(list(words), L, "cpu").numpy().astype("float32")
