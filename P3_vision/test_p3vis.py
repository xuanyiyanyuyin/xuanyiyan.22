# -*- coding: utf-8 -*-
"""
生成简单合成图, 测试 P3Vis 工业级视觉属性提取 + 离散映射
run: python test_p3vis.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from p3vis import P3Vis, VIS_DIM


def bg(size=128):
    return np.full((size, size, 3), 235, np.uint8)  # 浅灰背景


def red_circle(size=128):
    img = bg(size); cv2.circle(img, (size//2, size//2), size//3, (40, 40, 220), -1)  # BGR 红
    return img


def green_square(size=128):
    img = bg(size); cv2.rectangle(img, (size//4, size//4), (3*size//4, 3*size//4), (40, 200, 40), -1)
    return img


def blue_bar(size=128):
    img = bg(size); cv2.rectangle(img, (size//8, 2*size//5), (7*size//8, 3*size//5), (220, 60, 40), -1)
    return img


def yellow_rough(size=128):
    img = bg(size); cv2.circle(img, (size//2, size//2), size//3, (40, 200, 220), -1)  # 黄圆
    noise = np.random.randint(0, 70, (size, size, 3), np.uint8)                        # 噪声=粗糙纹理
    return cv2.add(img, noise)


TESTS = [
    ("红色圆  (期望: hue=red,   shape=round)",     red_circle()),
    ("绿色方块(期望: hue=green,  shape=irregular)", green_square()),
    ("蓝色横条(期望: hue=blue,   shape=elongated)", blue_bar()),
    ("黄色粗糙圆(期望: hue=yellow, texture=rough)",  yellow_rough()),
]


def main():
    p3v = P3Vis()
    print("=" * 70)
    print("P3Vis 工业级视觉属性测试 (合成图)")
    print("=" * 70)
    for name, img in TESTS:
        vec = p3v.process_image(img)   # 连续 256 维
        disc = p3v.to_discrete(img)    # 离散映射
        nz = int((np.abs(vec) > 1e-6).sum())
        layers = [("V1色彩", 0, 48), ("V2几何", 48, 112), ("V3纹理", 112, 160),
                  ("V4空间", 160, 208), ("V5语义", 208, 256)]
        seg_info = " ".join(f"{n}:{int((np.abs(vec[s:e])>1e-6).sum())}" for n, s, e in layers)
        print(f"\n【{name}】")
        print(f"  连续向量 256槽 激活{nz}个  [{seg_info}]")
        print(f"  离散属性: {disc}")
    print("\n" + "=" * 70)
    print("看点: 红圆→red/round, 绿方→green, 蓝条→blue/elongated, 黄粗糙→yellow/rough")
    print("=" * 70)


if __name__ == "__main__":
    main()
