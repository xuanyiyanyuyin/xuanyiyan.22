# -*- coding: utf-8 -*-
"""单组 golden 回归（arch 架构优化线，2026-11-11）

两把尺，都是单组数据（--max_sources 1 / bs8 / E60 / seed42）：
  default  = 台账 §0 原样           → 逐位回归底盘（50.09% / CE 2.0839 / 120 步）
  champion = 台账 §14 R16 冠军 argv → 敏感尺（各维坍缩量 / 已见水平 / 偏离档数）

★ 不认 [完成] 行里的 best_exam_acc：那是全局历史最佳回显，不是本轮读数
  （台账「假成功守卫」记过这个坑）。指纹只取 E60 判决行 + 训练前基线 + 实际走通步数。

用法:
  python regression.py --init                 # 从已有 golden 日志建指纹
  python regression.py --check               # 跑两把尺并逐项比对
  python regression.py --check --only default  # 快检（5s）
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"D:\ai\python_ltx25\Scripts\python.exe"
SCRIPT = "train_v22_cov_v1.py"
GOLDEN = os.path.join(HERE, "golden")
LOGS = os.path.join(HERE, "logs")

BASE = ["--use_generic", "0", "--gen_max", "0", "--exam_n_generic", "0",
        "--max_sources", "1", "--batch_size", "8", "--epochs", "60", "--seed", "42"]
CHAMP = BASE + ["--prompt_cond", "1", "--exam_singles", "1", "--exam_unseen_int", "1",
                "--train_vocab_cover", "1", "--regen_every", "1", "--int_map", "1",
                "--exam_unseen_shift", "1", "--spat_map", "1", "--roll_param", "1",
                "--exam_unseen_kind", "1", "--affine_map", "1", "--chan_perm", "1",
                "--chan_mult", "1"]
PROFILES = {"default": BASE, "champion": CHAMP, "champion_exact": CHAMP + ["--int_exact", "1"]}

SKIP = ("[日志]", "总耗时", "ETA", "GPU常驻", "i2i_m2_", "占用格数(>1%)=", "量程扫描")


def fingerprint(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or any(k in s for k in SKIP):
            continue
        if "D:\\" in s or "D:/" in s:
            continue
        # 接线自检行单独由 wiring_ok 判定，不进 sha（否则新增自检日志会伪装成数值回归失败）
        if ("E60" in s or "训练前基线" in s or "构造自检" in s) and re.search(r"\d", s):
            s = re.sub(r"\d+\.\d+s", "Xs", s)
            lines.append(s)
    m = re.search(r"本轮实际走通 (\d+) 步", text)
    steps = int(m.group(1)) if m else -1
    m2 = re.search(r"NaN跳步:(\d+)", text)
    nan = int(m2.group(1)) if m2 else -1
    body = "\n".join(lines)
    return {"sha": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
            "n_lines": len(lines), "steps": steps, "nan": nan,
            "wiring_ok": "[R18 接线自检] OK" in text}


def run(name, tag):
    out = os.path.join(LOGS, "arch_%s_%s.log" % (tag, name))
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        r = subprocess.run([PY, "-X", "utf8", SCRIPT] + PROFILES[name],
                           cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    fp = fingerprint(out)
    fp["exit"] = r.returncode
    fp["sec"] = round(time.time() - t0, 1)
    return fp, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", choices=["default", "champion", "champion_exact"], default=None)
    a = ap.parse_args()
    os.makedirs(GOLDEN, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    names = [a.only] if a.only else list(PROFILES)

    if a.init:
        src = {"default": "arch_R17_golden_default.log",
               "champion": "arch_R17_golden_champion.log",
               "champion_exact": "arch_init_champion_exact.log"}
        for n in names:
            p = os.path.join(LOGS, src[n])
            if not os.path.isfile(p):
                _, p = run(n, "init")
            fp = fingerprint(p)
            json.dump(fp, open(os.path.join(GOLDEN, n + ".json"), "w"), indent=2)
            print("[init] %-9s %s" % (n, fp))
        return 0

    if a.check:
        bad = 0
        for n in names:
            g = json.load(open(os.path.join(GOLDEN, n + ".json")))
            fp, out = run(n, "check")
            same = (fp["sha"] == g["sha"] and fp["steps"] == g["steps"]
                    and fp["nan"] == g["nan"] and fp["exit"] == 0)
            print("[check] %-9s %-4s sha %s/%s  steps %s/%s  %.0fs  %s" % (
                n, "OK" if same else "FAIL", fp["sha"], g["sha"],
                fp["steps"], g["steps"], fp["sec"], out))
            if not same:
                bad += 1
        print("RESULT: %s" % ("OK 逐位回归通过" if bad == 0 else "FAIL %d 把尺不符" % bad))
        return 1 if bad else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
