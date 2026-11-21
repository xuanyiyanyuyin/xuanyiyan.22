# -*- coding: utf-8 -*-
"""P3Code: 0参数规则式代码属性 (128D/字符) — 给V22 token的attr扩出代码模态。
槽位:
 0-7 字符类 one-hot: [字母,数字,下划线,运算符,括号,空白,'#/'(注释符),中文]
 8 行首缩进/8(clip1)  9 括号深度/8(clip1)  10 处于行内注释后  11 处于字符串内
 12 该行含中文  13 该行代码符号/字母占比  14-15 语言 one-hot [py, js]
 16-31 关键字 one-hot [def,class,return,if,else,for,while,import,from,print,self,None,True,False,function,const]
 32-127 预留0
"""
try:
    import torch
except Exception:
    torch = None

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


def code_np(item):
    import re as _re, numpy as np
    _KW = ['def', 'class', 'return', 'if', 'else', 'for', 'while', 'import', 'from', 'print', 'self', 'None', 'True', 'False', 'function', 'const']
    _OPS = [43,45,42,47,37,61,60,62,33,38,124,94,126]
    _BR = [40,41,91,93,123,125]
    _TOK = _re.compile(r"[A-Za-z_]+")

    words, L = item
    s = "".join(words)[:L]
    n = len(s)
    out = np.zeros((L, 128), dtype=np.float32)
    if n == 0:
        return out
    cp = np.fromiter((ord(c) for c in s), dtype=np.int32, count=n)
    alpha = ((cp >= 65) & (cp <= 90)) | ((cp >= 97) & (cp <= 122))
    digit = (cp >= 48) & (cp <= 57)
    under = cp == 95
    ops = np.isin(cp, _OPS)
    br = np.isin(cp, _BR)
    space = (cp == 32) | (cp == 9) | (cp == 10) | (cp == 13)
    hsh = (cp == 35) | (cp == 47)
    cjk = (cp >= 0x4E00) & (cp <= 0x9FFF)
    for i, m in enumerate((alpha, digit, under, ops, br, space, hsh, cjk)):
        out[:n, i] = m
    nl = np.where(cp == 10)[0]
    starts = np.concatenate(([0], nl + 1))
    ends = np.concatenate((nl, [n]))
    q = ((cp == 34) | (cp == 39)).astype(np.int32)
    for a, b in zip(starts, ends):
        line = s[a:b]
        ind = len(line) - len(line.lstrip(" "))
        out[a:b, 8] = min(ind / 8.0, 1.0)
        ci = line.find("#")
        cj = line.find("//")
        pos = [x for x in (ci, cj) if x >= 0]
        if pos:
            out[a + min(pos):b, 10] = 1.0
        out[a:b, 11] = np.cumsum(q[a:b]) % 2
        out[a:b, 12] = 1.0 if cjk[a:b].any() else 0.0
        Ln = max(b - a, 1)
        out[a:b, 13] = min(float((ops[a:b] | br[a:b]).sum()) / Ln, 1.0)
        out[a:b, 14] = 1.0 if ("def " in line or "import " in line or "self." in line) else 0.0
        out[a:b, 15] = 1.0 if ("function " in line or "const " in line or "=>" in line) else 0.0
    step = np.where(np.isin(cp, [40, 91, 123]), 1, np.where(np.isin(cp, [41, 93, 125]), -1, 0))
    depth = np.cumsum(step)
    out[:n, 9] = np.minimum(np.maximum(depth, 0) / 8.0, 1.0)
    for m in _TOK.finditer(s):
        t = m.group(0)
        if t in KW:
            out[m.start():m.end(), 16 + KW.index(t)] = 1.0
    return out
