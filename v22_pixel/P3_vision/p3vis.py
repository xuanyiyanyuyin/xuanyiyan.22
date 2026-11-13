# -*- coding: utf-8 -*-
"""
P3Vis — V22 工业级视觉属性栈 (白盒规则引擎, 0 可学习参数)
===========================================================
填 P3 属性向量的视觉区间 [128:384] (256D), 与语言区间 [0:128] 同层并列。

设计对齐 P3 语言侧的分层颗粒度:
  语言:  词类 → 语义 → 句法 → 逻辑 → 语用
  视觉:  V1色彩物理 → V2几何形态 → V3纹理材质 → V4空间深度 → V5时序语义

槽位布局 (全局绝对索引):
  V1 色彩物理层  128-175 (48)  [本模块相对 0-47]    — 规则可算, 已实现
  V2 几何形态层  176-239 (64)  [相对 48-111]        — 规则可算, 已实现
  V3 纹理材质层  240-287 (48)  [相对 112-159]       — 规则可算, 已实现
  V4 空间深度层  288-335 (48)  [相对 160-207]       — 部分需模型, 能算的已填
  V5 时序语义层  336-383 (48)  [相对 208-255]       — 需模型/多帧, 预留占位

原则(架构师定): 不做归一化, 原始特征值直填, 由下游多路 loss + 控制层自适应。
"""
import numpy as np
import cv2

try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# 视觉区间总维度 & 各层相对偏移
VIS_DIM = 256
V1_OFF, V1_LEN = 0, 48     # 色彩物理
V2_OFF, V2_LEN = 48, 64    # 几何形态
V3_OFF, V3_LEN = 112, 48   # 纹理材质
V4_OFF, V4_LEN = 160, 48   # 空间深度
V5_OFF, V5_LEN = 208, 48   # 时序语义


class P3Vis:
    """视觉属性规则引擎: image -> 256D 视觉属性向量 (填 P3 的 [128:384])"""

    def __init__(self, kmeans_k: int = 4, chroma: bool = False, kmeans_seed=None):
        self.kmeans_k = kmeans_k
        # [色度补槽] chroma=True 才填 V1:44-47 与 V3:34-47 这 18 个原本恒零的槽。
        # 默认 False ⇒ 向量逐位与旧版相同(旧口径可复现); 打开后 P3Vis 输出内容变化 ⇒ 与旧分数不可比, 属新口径。
        self.chroma = bool(chroma)
        # [可复现性] cv2.kmeans 走 OpenCV 内部 RNG, 旧代码从未播种 ⇒ V1:28-43 这 16 个槽
        # 同一张图每次调用都重抽(实测最大差 0.44), 连同一 run 内部都不自洽。给定 seed 后逐次可复现。
        self.kmeans_seed = kmeans_seed
        self.tex_backend = "skimage" if _HAS_SKIMAGE else "MISSING"

    # ============================================================
    # 主入口
    # ============================================================
    def process_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """输入 BGR 图 (cv2 格式), 返回 256D 视觉属性向量。"""
        vec = np.zeros(VIS_DIM, dtype=np.float32)
        if img_bgr is None or img_bgr.size == 0:
            return vec

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fg_mask, main_cnt = self._foreground(gray)

        vec[V1_OFF:V1_OFF + V1_LEN] = self._v1_color(img_bgr)
        vec[V2_OFF:V2_OFF + V2_LEN] = self._v2_geometry(gray, fg_mask, main_cnt)
        vec[V3_OFF:V3_OFF + V3_LEN] = self._v3_texture(gray)
        vec[V4_OFF:V4_OFF + V4_LEN] = self._v4_space_depth(gray, fg_mask, main_cnt)
        vec[V5_OFF:V5_OFF + V5_LEN] = self._v5_temporal_semantic(img_bgr)
        if self.chroma:
            # 只写空槽, 不动已实现的槽 ⇒ 关掉开关时与旧版逐位相同
            self._chroma_fill(img_bgr, vec[V1_OFF:V1_OFF + V1_LEN], vec[V3_OFF:V3_OFF + V3_LEN])
        return vec

    # ============================================================
    # 离散映射: 连续特征 → 离散视觉值 (供 P3-L cross_vl 跨模态关联)
    # 双流互补: process_image 连续256维进 ABC(精度);
    #           to_discrete 离散值进 P3-L(关联/看图说话)。
    # ============================================================
    def to_discrete(self, img_bgr) -> dict:
        d = {}
        if img_bgr is None or img_bgr.size == 0:
            return d
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        mask = sat > 30
        fg = hsv[mask] if mask.any() else hsv.reshape(-1, 3)

        # --- 色相 + 冷暖 ---
        h_deg = float(np.median(fg[:, 0])) * 2 if len(fg) else 0.0  # H 0-180 → 0-360
        if h_deg < 15 or h_deg > 330:   hue = "red"
        elif h_deg < 45:                hue = "orange"
        elif h_deg < 75:                hue = "yellow"
        elif h_deg < 165:               hue = "green"
        elif h_deg < 195:               hue = "cyan"
        elif h_deg < 270:               hue = "blue"
        else:                           hue = "magenta"
        d["vis_hue"] = hue
        d["vis_warmth"] = "warm" if (h_deg < 60 or h_deg > 300) else "cool"

        # --- 亮度 / 饱和 ---
        v_mean = float(hsv[:, :, 2].mean()) / 255
        d["vis_brightness"] = "bright" if v_mean > 0.6 else ("dark" if v_mean < 0.3 else "medium")
        s_mean = float(sat.mean()) / 255
        d["vis_saturation"] = "vivid" if s_mean > 0.4 else ("dull" if s_mean < 0.15 else "moderate")

        # --- 形状 / 对称 / 位置 (前景轮廓) ---
        _, fgm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(fgm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt) + 1e-6
            peri = cv2.arcLength(cnt, True) + 1e-6
            circ = 4 * np.pi * area / peri ** 2
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / max(bh, 1)
            if circ > 0.7:              d["vis_shape"] = "round"
            elif ar > 1.8 or ar < 0.55: d["vis_shape"] = "elongated"
            else:                       d["vis_shape"] = "irregular"
            d["vis_symmetry"] = "symmetric" if 0.8 < ar < 1.25 else "asymmetric"
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
                h_, w_ = gray.shape
                col = ["left", "center", "right"][min(2, int(cx / w_ * 3))]
                row = ["top", "center", "bottom"][min(2, int(cy / h_ * 3))]
                d["vis_position"] = "center" if (col == "center" and row == "center") else f"{row}-{col}"

        # --- 纹理 (LBP 主峰) ---
        if _HAS_SKIMAGE:
            lbp = local_binary_pattern(cv2.resize(gray, (128, 128)), P=8, R=1, method="uniform")
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            peak = int(np.argmax(hist))
            d["vis_texture"] = "smooth" if peak < 3 else ("gradient" if peak < 6 else "rough")

        return d

    # ============================================================
    # 前景分离 (供 V2/V4 共用)
    # ============================================================
    def _foreground(self, gray):
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main = max(cnts, key=cv2.contourArea) if cnts else None
        return fg, main

    # ============================================================
    # V1 色彩物理层 (48) — 主色/曝光/色温/心理/主色块
    # ============================================================
    def _v1_color(self, bgr):
        v = np.zeros(V1_LEN, dtype=np.float32)
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # 0-2 主色调 HSV 均值
        v[0:3] = hsv.reshape(-1, 3).mean(0) / np.array([180, 255, 255])
        # 3-5 主色 Lab 均值
        v[3:6] = lab.reshape(-1, 3).mean(0) / 255.0
        # 6-7 色彩方差 / 丰富度(colorfulness, Hasler-Süsstrunk)
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        rg, yb = R - G, 0.5 * (R + G) - B
        v[6] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
        v[7] = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) + v[6]
        # 8-10 H通道直方图 偏度/峰度/熵
        hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
        hist_h = hist_h / (hist_h.sum() + 1e-8)
        mu = (hist_h * np.arange(18)).sum()
        sig = np.sqrt(((np.arange(18) - mu) ** 2 * hist_h).sum()) + 1e-8
        v[8] = float(((np.arange(18) - mu) ** 3 * hist_h).sum() / sig ** 3)  # 偏度
        v[9] = float(((np.arange(18) - mu) ** 4 * hist_h).sum() / sig ** 4)  # 峰度
        v[10] = float(-(hist_h * np.log2(hist_h + 1e-8)).sum())              # 色相熵

        # === 光学与曝光 11-18 ===
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        v[11] = float(Y.mean() / 255)                       # 全局亮度
        v[12] = float(Y.std() / 128)                        # 对比度
        v[13] = float((Y.max() - Y.min()) / 255)            # 动态范围
        # 曝光均匀度: 4x4 分块亮度方差的逆
        blk = cv2.resize(Y, (4, 4)).flatten()
        v[14] = float(1.0 / (1.0 + blk.std()))
        v[15] = float((Y > 250).mean())                     # 高光溢出率
        v[16] = float((Y < 5).mean())                       # 暗部死黑率
        v[17] = float(((Y > 5) & (Y < 250)).mean())         # 有效动态占比
        v[18] = float(np.median(Y) / 255)                   # 亮度中位

        # === 色温与白平衡 19-23 ===
        r_avg, g_avg, b_avg = R.mean(), G.mean(), B.mean()
        v[19] = float((r_avg - b_avg) / 255)                # 暖冷偏移(色温代理)
        v[20] = float((g_avg - 0.5 * (r_avg + b_avg)) / 255)  # 品绿偏移(tint)
        v[21] = float(r_avg / (b_avg + 1e-6))               # R/B 比(白平衡)
        v[22] = float(hsv[:, :, 1].mean() / 255)            # 全局饱和
        v[23] = float((hsv[:, :, 1] > 30).mean())           # 有效彩色占比

        # === 色彩心理 24-27 ===
        H = hsv[:, :, 0]
        warm = ((H < 30) | (H > 150)).mean()
        cool = ((H >= 60) & (H <= 150)).mean()
        v[24] = float(warm)
        v[25] = float(cool)
        v[26] = float(warm - cool)                          # 冷暖倾向
        v[27] = float(hsv[:, :, 1].std() / 128)             # 饱和活跃度

        # === K-Means Top-4 主色块 28-43 (4色 × [H,S,V,占比]) ===
        try:
            Z = hsv.reshape(-1, 3).astype(np.float32)
            if len(Z) > self.kmeans_k:
                if self.kmeans_seed is not None:
                    cv2.setRNGSeed(int(self.kmeans_seed) & 0x7FFFFFFF)   # 每次调用前重置 ⇒ 与调用次序无关
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(Z, self.kmeans_k, None, crit, 2,
                                                cv2.KMEANS_PP_CENTERS)
                counts = np.bincount(labels.flatten(), minlength=self.kmeans_k)
                order = np.argsort(-counts)
                for i, idx in enumerate(order[:4]):
                    c = centers[idx] / np.array([180, 255, 255])
                    v[28 + i * 4:28 + i * 4 + 3] = c
                    v[28 + i * 4 + 3] = counts[idx] / len(labels)
        except Exception:
            pass
        # 44-47 预留(色彩和谐度等)
        return v

    # ============================================================
    # V2 几何形态层 (64) — 包围盒/边缘/Hu矩/拓扑
    # ============================================================
    def _v2_geometry(self, gray, fg, cnt):
        v = np.zeros(V2_LEN, dtype=np.float32)
        h, w = gray.shape

        # --- 基础图元与包围盒 0-15 ---
        v[0] = float((fg > 0).mean())                       # 前景占比
        if cnt is not None and cv2.contourArea(cnt) > 5:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            v[1:5] = [x / w, y / h, bw / w, bh / h]         # bbox 归一坐标
            hull = cv2.convexHull(cnt)
            harea = cv2.contourArea(hull) + 1e-6
            v[5] = float(area / harea)                       # solidity 凸包比
            v[6] = float(bw / max(bh, 1))                    # 长宽比
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            v[7] = float(max(rw, rh) / (min(rw, rh) + 1e-6)) # 最小外接矩形长宽比
            v[8] = float(area / (bw * bh + 1e-6))            # extent 填充度
            peri = cv2.arcLength(cnt, True) + 1e-6
            v[9] = float(4 * np.pi * area / peri ** 2)       # 圆度
            v[10] = float(peri ** 2 / (4 * np.pi * area + 1e-6))  # 紧凑度

            # --- 轮廓与边缘 16-31 ---
            edges = cv2.Canny(gray, 50, 150)
            v[16] = float((edges > 0).mean())               # 边缘密度
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            ang = (np.arctan2(gy, gx) + np.pi)               # 0..2pi
            mag = np.sqrt(gx ** 2 + gy ** 2)
            hog, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi), weights=mag)
            hog = hog / (hog.sum() + 1e-8)
            v[17:25] = hog                                   # HOG 8方向
            v[25] = float(mag.mean() / 255)                  # 平均梯度强度
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            v[26] = float(len(approx))                       # 多边形边数
            v[27] = float(cv2.isContourConvex(approx))       # 是否凸

            # --- Hu矩 + 形状描述子 32-47 ---
            hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # 对数标度
            v[32:39] = hu / 30.0                              # 7个不变矩
            v[39] = float(bw / max(bh, 1))                    # 矩形度代理

            # --- 拓扑与连通性 48-63 ---
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg)
            v[48] = float(n_labels - 1)                       # 连通域数(去背景)
            if n_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                v[49] = float(areas.max() / (h * w))          # 最大连通域占比
            inv = cv2.bitwise_not(fg)
            n_holes, _, _, _ = cv2.connectedComponentsWithStats(inv)
            v[50] = float(max(0, n_holes - 2))                # 孔洞数(近似)
            v[51] = float((n_labels - 1) - max(0, n_holes - 2))  # 欧拉数近似
        return v

    # ============================================================
    # V3 纹理材质层 (48) — GLCM/LBP/材质/缺陷
    # ============================================================
    def _v3_texture(self, gray):
        v = np.zeros(V3_LEN, dtype=np.float32)
        g_small = cv2.resize(gray, (128, 128))

        # --- GLCM 0-15 (4指标 × 4方向) ---
        # [口径纪律] 0-23 整段依赖 skimage; 缺席时原本会静默留零(不报错不打字),
        # 于是"纹理层在场"变成机器相关的隐藏口径轴。宁缺不假: 由调用方在缺包时拒绝启动。
        if _HAS_SKIMAGE:
            q = (g_small // 32).astype(np.uint8)  # 量化到8级
            glcm = graycomatrix(q, distances=[1],
                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=8, symmetric=True, normed=True)
            for i, prop in enumerate(['contrast', 'energy', 'homogeneity', 'correlation']):
                vals = graycoprops(glcm, prop).flatten()  # 4方向
                v[i * 4:i * 4 + 4] = vals / (vals.max() + 1e-8) if prop == 'contrast' else vals

        # --- LBP 16-23 ---
        if _HAS_SKIMAGE:
            lbp = local_binary_pattern(g_small, P=8, R=1, method='uniform')
            hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
            v[16] = float(np.argmax(hist)) / 10              # 主峰位置
            v[17] = float(hist.max())                        # 主峰强度
            v[18] = float(-(hist * np.log2(hist + 1e-8)).sum())  # 纹理熵(粗糙度)
            v[19:24] = hist[:5]                              # 直方图前5

        # --- 材质物理估计 24-31 (简化) ---
        v[24] = float((g_small > 240).mean())               # 高光/镜面反射代理
        v[25] = float(g_small.std() / 128)                  # 表面粗糙度代理
        v[26] = float(g_small.mean() / 255)                 # 漫反射率代理(反照率)
        lap = cv2.Laplacian(g_small, cv2.CV_32F)
        v[27] = float(lap.var() / 1000)                     # 清晰度/细节丰富度

        # --- 规律性与缺陷 32-47 ---
        f = np.fft.fftshift(np.abs(np.fft.fft2(g_small)))
        v[32] = float(f.std() / (f.mean() + 1e-6))          # 频域能量分散(周期性)
        gx = cv2.Sobel(g_small, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g_small, cv2.CV_32F, 0, 1)
        v[33] = float(np.abs(gx).mean() / (np.abs(gy).mean() + 1e-6))  # 纹理方向性
        return v

    # ============================================================
    # 色度补槽 (chroma=True): V1:44-47 白平衡 + V3:34-47 色度分布与跨通道结构
    # 动机: 原实现的色彩量只有均值(V1:3-5 的 Lab 均值)与标量比(V1:19-21),
    #       没有任何"色度分布/跨通道相关"的量; 而上色任务的监督恰恰在色度分布上。
    # 尺度沿用本文件既有习惯(/255、/128), 不额外归一(架构师原则: 原始特征值直填)。
    # ============================================================
    def _chroma_fill(self, bgr, v1, v3):
        f = bgr.astype(np.float32) / 255.0
        B_, G_, R_ = f[:, :, 0], f[:, :, 1], f[:, :, 2]

        # === V1 44-47: Gray-world 白平衡增益(原本只有 R/B 标量比 :182, 没有闭式增益) ===
        m = np.array([R_.mean(), G_.mean(), B_.mean()], dtype=np.float64) + 1e-6
        gain = m.mean() / m
        v1[44] = float(gain[0] - 1.0)
        v1[45] = float(gain[1] - 1.0)
        v1[46] = float(gain[2] - 1.0)
        v1[47] = float(gain.max() / gain.min() - 1.0)          # 增益跨度 = 色偏强度

        # === V3 34-39: 色度平面(Lab 的 a/b, YCrCb 的 Cr/Cb)的扩展与形态, 原来只有均值 ===
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        a, bb = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
        cr, cb = ycc[:, :, 1] - 128.0, ycc[:, :, 2] - 128.0
        v3[34] = float(cr.std() / 64.0)
        v3[35] = float(cb.std() / 64.0)
        v3[36] = float(a.std() / 64.0)
        v3[37] = float(bb.std() / 64.0)
        v3[38] = float(((a - a.mean()) ** 3).mean() / (a.std() ** 3 + 1e-6))   # a 偏度
        v3[39] = float(((bb - bb.mean()) ** 3).mean() / (bb.std() ** 3 + 1e-6))

        # === V3 40-43: 跨通道相关结构(全仓原本没有任何 cov/corrcoef) ===
        # [2026-11-04 缺陷修复] np.corrcoef 内部按 stddev 相除, 恒定通道(纯色照片/纯色patch)会
        # 让 V3:40-42 变成 NaN 并被烤进训练特征(实测 RuntimeWarning _function_base_impl.py:3000)。
        # 改成手写相关: 协方差除以各自标准差, 标准差≈0 时该格协方差本来就是 0 ⇒ 相关记 0, 不再产生 NaN。
        stack = np.vstack([R_.ravel(), G_.ravel(), B_.ravel()])
        cov3 = np.cov(stack)
        sd3 = np.sqrt(np.clip(np.diag(cov3), 0.0, None))
        cc = cov3 / np.outer(np.where(sd3 < 1e-9, 1.0, sd3), np.where(sd3 < 1e-9, 1.0, sd3))
        v3[40] = float(cc[0, 1])
        v3[41] = float(cc[0, 2])
        v3[42] = float(cc[1, 2])
        ev = np.linalg.eigvalsh(np.cov(stack) + 1e-9 * np.eye(3))
        v3[43] = float(ev[-1] / (ev.sum() + 1e-9))              # 主色方向占比(色彩自由度)

        # === V3 44-47: Cr×Cb 粗二维直方图(暖亮/暖暗/冷亮/冷暗四象限占比) ===
        q1 = float(((cr > 0) & (cb > 0)).mean())
        q2 = float(((cr > 0) & (cb <= 0)).mean())
        q3 = float(((cr <= 0) & (cb > 0)).mean())
        q4 = float(((cr <= 0) & (cb <= 0)).mean())
        v3[44], v3[45], v3[46], v3[47] = q1, q2, q3, q4

    # ============================================================
    # V4 空间深度层 (48) — 能算的填(方位/显著性), 深度/透视留 TODO
    # ============================================================
    def _v4_space_depth(self, gray, fg, cnt):
        v = np.zeros(V4_LEN, dtype=np.float32)
        h, w = gray.shape

        # 0-2 深度粗估(TODO: 接单目深度模型; 暂用聚焦/模糊梯度做近远景代理)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        focus = np.abs(lap)
        v[0] = float(focus.mean() / 100)                    # 全局清晰度(近景代理)
        v[1] = float(focus.std() / 100)                     # 景深方差代理
        # 3-7 透视(TODO: 消失点检测, 留空)

        # 16-31 空间方位: 前景质心落在九宫格哪格 (one-hot)
        if cnt is not None and cv2.contourArea(cnt) > 5:
            M = cv2.moments(cnt)
            cx, cy = M['m10'] / (M['m00'] + 1e-6), M['m01'] / (M['m00'] + 1e-6)
            gx_, gy_ = min(2, int(cx / w * 3)), min(2, int(cy / h * 3))
            v[16 + gy_ * 3 + gx_] = 1.0                      # 九宫格 one-hot
            v[25] = float(cx / w)                            # 质心 x
            v[26] = float(cy / h)                            # 质心 y

        # 32-39 显著性(用 spectral residual, 需 opencv-contrib; 无则跳过)
        try:
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, smap = sal.computeSaliency(gray)
            if ok:
                smap = (smap * 255).astype(np.uint8)
                v[32] = float(smap.mean() / 255)             # 显著性均值
                v[33] = float(smap.max() / 255)              # 显著性峰值
                my, mx = np.unravel_index(np.argmax(smap), smap.shape)
                v[34], v[35] = float(mx / w), float(my / h)  # 显著点坐标
                v[36] = float((smap > 128).mean())           # 显著区占比
        except Exception:
            pass  # opencv-contrib 未装则 V4显著性留空
        return v

    # ============================================================
    # V5 时序/运动/高层语义 (48) — 需模型/多帧, 预留占位
    # ============================================================
    def _v5_temporal_semantic(self, bgr):
        v = np.zeros(V5_LEN, dtype=np.float32)
        # 0-15  运动光流   TODO: 需多帧输入
        # 16-31 场景分类   TODO: 需场景分类模型(室内/外, 天气)
        # 32-47 跨模态锚点 TODO: OCR/标志检测 → 视觉-文本强对齐
        # 能免费算的: 场景复杂度(边缘熵) 放 v[16]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        v[16] = float((edges > 0).mean())                    # 场景复杂度(边缘密度)
        return v


# ================================================================
# 独立验证: python p3vis.py <图片路径>
# ================================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_apple.bmp"
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] 无法读取图片: {path}")
        sys.exit(1)

    p3v = P3Vis()
    vis = p3v.process_image(img)  # 256D 视觉属性

    # 拼成完整 384D P3 向量(语言区 0-127 此处留空, 因为输入是纯图)
    full = np.zeros(384, dtype=np.float32)
    full[128:384] = vis

    print("=" * 64)
    print(f"P3Vis 工业级视觉属性 Forward Pass  |  图: {path}  |  shape: {img.shape}")
    print("=" * 64)
    layers = [("V1 色彩物理", 0, 48), ("V2 几何形态", 48, 112),
              ("V3 纹理材质", 112, 160), ("V4 空间深度", 160, 208),
              ("V5 时序语义", 208, 256)]
    for name, s, e in layers:
        seg = vis[s:e]
        nz = int((np.abs(seg) > 1e-6).sum())
        print(f"  [{name}] 槽位 {128+s:>3}-{128+e-1:<3} | 激活 {nz:>2}/{e-s} | "
              f"范围[{seg.min():+.3f},{seg.max():+.3f}] 均值{seg.mean():+.3f}")
    total_nz = int((np.abs(vis) > 1e-6).sum())
    print("-" * 64)
    print(f"  视觉区间 [128:384] 共 256 槽, 激活 {total_nz} 个")
    print(f"  完整 P3 向量 shape = {full.shape} (语言0-127留空 + 视觉128-383)")
    print(f"  skimage(GLCM/LBP) = {'可用' if _HAS_SKIMAGE else '缺失, V3部分留空'}")
    print("=" * 64)
