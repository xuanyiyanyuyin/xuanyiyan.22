"""
═══════════════════════════════════════════════════════════════
V22.1 图生图 M2 — 64×64 patch化 + P7 跨patch路由 (20261024)
═══════════════════════════════════════════════════════════════

架构 (规划见 D:/ai/V22.1_图生图M2_规划_20261024.md):
  64×64 退化图 → 16 个 16×16 patch
  每 patch: 768 bins → 冻结bin_embed查表 + 3×3/5×5 均值上下文  (M1 v4.1 机制, batch=b×16)
  P7 跨patch路由: patch_token(VisualEmbed均池化) 自注意力 4×4 栅格 → word_out
    → W_pctx 投影加到对应 patch 全部 768 位                    ← vision 接入引擎第一步
  全局身份 h: sent_vec(P7) + P3Vis(整图) → input_proj → encoder → 球面×h_scale
  逐位门控 hp = h ⊙ tanh(clamp(gate(inpatch_pos),±2))          (v4.1)
  logits = (feat + pctx + hp + patch_pos + inpatch_pos)·bin_embedᵀ → CE(12288位×16阶)
  位置编码因式分解: patch_pos[16,256] + inpatch_pos[768,256]
  局部给均值矩阵, 全局给 P7 —— 跨patch 12288² 均值矩阵不可行也无必要

显存: ~85MB/样本 → batch 64 ≈ 5.4GB 纯显存甜点 (= M1 b1024 的有效规模)
精度: FP32 全程; 分配器: 默认(超额响亮OOM, 不静默溢出)

用法:
  python train_v22_m2.py --epochs 1200               # 正式
  python train_v22_m2.py --smoke                     # 冒烟
"""
import os
# [E5修复] 防碎片分配参数(垃圾回收阈值+分段上限), 对抗 regen 双份缓存+桌面挤压的碎片化OOM;
# 仍不用 expandable_segments(静默溢出共享显存), 超额依旧响亮OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "garbage_collection_threshold:0.8,max_split_size_mb:256")
import torch, torch.nn as nn, torch.nn.functional as F
import sys, glob, time, random, argparse
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import DEVICE, SAVE_DIR, BASE_DIR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "P3_vision"))
from p3vis import P3Vis
from P7_cross_sent.model import P7WordRouter2048

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.012)
parser.add_argument("--n_bins", type=int, default=16)
parser.add_argument("--use_generic", type=int, default=1)
parser.add_argument("--k_labeled", type=int, default=16)
parser.add_argument("--k_generic", type=int, default=4)
parser.add_argument("--gen_max", type=int, default=300)
parser.add_argument("--exam_n_generic", type=int, default=30)
parser.add_argument("--regen_every", type=int, default=100)
parser.add_argument("--exam_every", type=int, default=100)
parser.add_argument("--clip_norm", type=float, default=5.0)
parser.add_argument("--v2_probe_every", type=int, default=0,
                    help="非0时: 每N轮的首个batch打印 v2 分支级探针(路由熵/支路权重/copy门g/合成与copy的logits尺度/用到几个类)")
parser.add_argument("--seam_probe_every", type=int, default=0,
                    help="非0时: 每N轮首个batch打印无缝邻域探针(注入增益/方向选择熵/各方向平均权重/seam幅度)")
parser.add_argument("--v2_attn_gain", type=float, default=1.0,
                    help="v2 注意力残差的初始增益(可学)。1.0=原口径(实测E2即炸到 std43→合成140→g塌0); 建议 0.1/0.02")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--exam_chunk", type=int, default=24,
                    help="[R20] 考试分块大小(原写死24); v2 或低显存时调小, 24为默认 ⇒ 逐位不变")
parser.add_argument("--vram_cap", type=float, default=0.0,
                    help="[R20] 本进程显存上限GB(0=沿用环境变量 M2_VRAM_CAP_GB, 默认6.0)")
parser.add_argument("--sigma_curriculum", type=int, default=0, help="E4课程开关: 1=三段σ课程, 0=全档(E2最佳配置)")
parser.add_argument("--max_sources", type=int, default=0, help="诊断用: 只用前N张带标签源图(0=全部)")
parser.add_argument("--gate_pair", type=int, default=1, help="[缺陷修复] 身份门控吃(patch,位内位置)交互: 1=修复版, 0=旧版纯inpatch")
parser.add_argument("--encoder_width", type=int, default=256, help="N2: 身份encoder隐层宽度(256=原版)")
parser.add_argument("--spatial_upgrade", type=int, default=0, help="[空间升级] 1=方向线状上下文+patch内邻域注意力, 0=原版固定均值")
parser.add_argument("--gate_task", type=int, default=0,
                    help="1=copy/transform 门加逐任务偏置(task_id 来自数据管道), 治疗一个全局门同时服务去噪/遮补/上色")
parser.add_argument("--s_seamless", type=int, default=0,
                    help="[自学习·窄融合] 在 S 路径上加跨patch无缝邻域支路, 值=全局偏移个数(4/8/16); "
                         "混合权重由 router 按内容自学, 注入有可学增益(初值0.1)。需 --spatial_upgrade 1")
parser.add_argument("--transform_v2", type=int, default=0, help="[转换多样性v2] 1=无缝全图邻域+自学习路由+多头注意力+copy/transform门控")
parser.add_argument("--v2_sdpa", type=int, default=0,
                    help="[R24/P1] 1=v2 的 4 头 patch 内注意力走 F.scaled_dot_product_attention(不物化 768² 分数矩阵, "
                         "bs8/chunk24 全量口径放得下); 0=旧显式矩阵路(逐位原口径)。跑前有旧路 vs SDPA 等价闸(≤1e-4)")
parser.add_argument("--task_cond", type=int, default=0,
                    help="[自学习T1] §15-1: 任务嵌入+σ嵌入+mask摘要 加进全局h (三者零初始化, 初始行为与关闭时逐位一致)")
parser.add_argument("--pure_learn_ops", type=int, default=0,
                    help="[清一色自学习PL] 1=S路上所有固定算子换成可学, 初值精确等于冠军口径: "
                         "(1)avg3/avg5/shiftM 由冻结buffer变成L1行归一化可学核; "
                         "(2)14个常数1.0加法换成吃(位内内容+任务/σ/mask条件)的逐支路门 1+tanh(·), 初值恒1.0; "
                         "(3)gate_pair 原本只吃位置(源码注释'gp 输入无关'), 额外吃条件。需 --spatial_upgrade 1 --gate_pair 1")
parser.add_argument("--color_aux", type=float, default=0.0,
                    help="[色度监督] 从 softmax 解出期望 RGB → YCbCr, 用干净目标图的 Cb/Cr 做 L1 辅助损失。"
                         "动机: 现在 loss 是纯逐 bin CE(:652), 上色任务全程没有任何色彩监督, "
                         "等权 CE 把'亮度对/颜色错'判成基本没错(实测真实池上色 D 比 C 还低 3.31)。"
                         "默认 0=不加(损失逐位与旧版相同)。量级实测: 把 R/B 整个交换才 aux=0.029, 而 CE≈1.0 "
                         "⇒ 权重需 10~30 才与 CE 同量级; 取 1~3 会因这项太小而测出假阴性(判据: 起跑前先看纯CE与aux两项的比值)")
parser.add_argument("--p3v_allow_no_skimage", type=int, default=0,
                    help="承认「本机无 skimage ⇒ 纹理层 V3:0-23 恒零」这一口径强行跑(默认拒绝启动)。"
                         "用它就等于声明: 该 run 的纹理层不在场, 不可与有 skimage 的 run 比")
parser.add_argument("--p3v_chroma", type=int, default=0,
                    help="[色度补槽] 1=P3Vis 把 V1:44-47 与 V3:34-47 这 18 个恒零槽填上色度量"
                         "(Gray-world 白平衡增益 / Lab a与b 及 YCrCb Cr与Cb 的扩展与偏度 / 跨通道相关与协方差主轴占比 / Cr×Cb 四象限直方图)。"
                         "默认 0 ⇒ 向量逐位与旧版相同; 打开后 P3Vis 输出内容变化, 与旧分数不可比(新口径)")
parser.add_argument("--gate_span", type=float, default=1.0,
                    help="[PL量程] 逐支路门 = 1 + span·tanh(·), 即值域 (1-span, 1+span)。默认 1.0=(0,2)＝B/D 的口径; "
                         "实测真实池 D 有 9 个(任务×通道)组合顶到上限 2.0 ⇒ 量程成了硬约束, 放宽试 2.0=(0,4)。初值恒 1.0 与量程无关")
parser.add_argument("--exam_only", type=str, default="",
                    help="[只评测] 载入指定存点跑一次考试(含按任务读门), 不训练不建训练集; "
                         "存点自带 args ⇒ 会逐项比对口径, 级别不符只告警不静默")
parser.add_argument("--exam_combo_all", type=int, default=0,
                    help="[G12尺子] 1=考卷每张图配齐全部 7 个二元组合(基础行固定用第 1 个, 其余 6 个追加)。"
                         "目的: 把组合侧每个变形的 n 从 1 张图提到 7 张图——G8/G11 都因 n=1 作废过分型结论, "
                         "不先补这一格就没资格说'组合没连乘'。追加行不消耗随机数 ⇒ 原有各行输入/目标逐位不变")
parser.add_argument("--exam_manifest", type=int, default=1,
                    help="[G12配对] 1=把考卷每一行的(行号,图名,任务,变形名)写成 img2img_out_m2/exam_manifest.csv, "
                         "让外部探针按行表配对而不是按 i//3、i//8 猜(那种猜法在 G11 造成过 4 格误配)")
parser.add_argument("--exam_singles", type=int, default=0,
                    help="[G8对照格] 1=考卷在 7 个未见二元组合之外, 每张图再追加 5 个**单元形**格(偏红/偏绿/变暗/翻转/平移各一)。"
                         "判的是缺口在元件还是在组合装配: 单元形过线而组合不过 ⇒ 组合装配; 单元形也不过 ⇒ 元件本身。"
                         "追加格不消耗随机数 ⇒ 原 21 对的输入/目标逐位不变; 但考卷行数 21→56 ⇒ 总 pixel_acc 与历史不同口径(分格仍可比)")
parser.add_argument("--cont_loss", type=float, default=0.0,
                    help="[F3 覆盖修复] 期望值→格中心 的全通道 L1 回归辅助损失权重(0=关); 治 CE-over-bins 丢弃序数/连续信息")
parser.add_argument("--comp_gate", type=int, default=0,
                    help="[R4 覆盖修复] 空间算子门改逐算子独立sigmoid(可组合, pair=两门同开); 0=旧加性logit(单元形训出+-, pair相消→0.5)")
parser.add_argument("--spat_init", type=float, default=0.0,
                    help="[R5 覆盖修复] refl/roll 线性层用小正交初值(=该值)点亮空间支路; 0=旧零初值(惰性)")
parser.add_argument("--exam_unseen_int", type=int, default=0,
                    help="[R6 尺子] 1=考卷追加**词表外强度**格: 变形名照旧(指示向量与训练时逐位相同), 但目标乘子换成 --unseen_mults 的值。"
                         "这一格量的是主人那条命题「规则覆盖不到后面模型能力就是死的」: 词表只有 5 个名字、常数写死 0.55/0.7 ⇒ "
                         "指示里根本没有强度这一位, 模型不可能知道。只加考试行不改结构/损失/权重; 追加行不消耗随机数 ⇒ 原行逐位不变。"
                         "总 pixel_acc 会因行数变化换口径, 故同时打印 base 行(原21对同口径)的分数, 跨轮比较只认 base 行与 [词表外强度] 两块")
parser.add_argument("--unseen_mults", type=str, default="0.40,0.70",
                    help="[R6 尺子] 词表外强度格的乘子列表(逗号分隔)。默认 0.40/0.70: 都在训练用过的 0.55 两侧, "
                         "且 0.70 与'变暗'的常数同值不同名 ⇒ 顺带量'同乘子换个名字'会不会被当成另一件事")
parser.add_argument("--train_vocab_cover", type=int, default=0,
                    help="[R7 词表采样] 1=训练集上色行的变形名改成按名序轮转(0,1,2,3,4,0,…)而不是 iid 抽。"
                         "B1 实测: 单组课程下训练集只 16 对、上色仅 5 条, 抽定后 60 轮复用 ⇒ 计数={偏红1,偏绿1,变暗1,翻转2,平移0}, "
                         "平移一次没训到(‖prompt_e‖[平移]=0.000 对上)。轮转保证 5/5 名各至少 1 条; "
                         "仍照抽一次 rng 以对齐随机流 ⇒ 非上色行的输入/目标逐位不变")
parser.add_argument("--int_cond", type=int, default=0,
                    help="[R8/F2 连续参数指示] 1=指示通道从'枚举名'升级为'名×强度': 送进 prompt_e 的向量不再是 0/1 multi-hot, "
                         "而是把激活的颜色名那一维换成该行的实际乘子 m(几何名保持 1.0)。同时训练侧上色行的 m 不再写死 MULT_TABLE, "
                         "改在 --train_int_range 内抽 ⇒ 强度这一位第一次有梯度。考试卷**逐位不变**(base/单元形仍用 MULT_TABLE, "
                         "词表外格仍用 0.40/0.70, 只是现在这个 m 会被送进指示通道) ⇒ 坍缩量可直接与 B2/B3 比。"
                         "0=旧口径(multi-hot 原样), 此时新增的 pval 张量与 prompts 逐位相同")
parser.add_argument("--train_int_range", type=str, default="0.35,0.85",
                    help="[R8] 训练侧强度抽样区间(逗号分隔下/上界)。默认 0.35~0.85 把考试用的 0.40/0.55/0.70/0.85 全包住, "
                         "并跨 MULT_TABLE 的两个写死常数 ⇒ 学的是'读 m', 不是'背两个常数'")
parser.add_argument("--chan_mult", type=int, default=0,
                    help="[R16 颜色规则参数化] 1=逐通道乘子也做成参数族: 指示向量尾部再带 3 个数 (mR,mG,mB), "
                         "档轴图按通道各自构造 M[b,c]=P(m_c)(与 R9 的名族**共用同一处插值实现** _Pm, 不新造第二套)。"
                         "⇒ '暖调(1.0,0.85,0.70)'、'白平衡偏冷'这类规则不再是词表外的东西: 偏红/偏绿/变暗 只是这个"
                         "三维空间里的三个点。跑前阳性对照写死: 给 (1,0.55,0.55) 且不给名 ⇒ M 必须与'请求偏红、强度0.55'"
                         "**逐位相同**(新族必须包含老预设, 与 R14 仿射包含镜像/roll 同一口径)。0=旧口径(R15 行为逐位不变)")
parser.add_argument("--chan_perm", type=int, default=0,
                    help="[R15 颜色种类参数化] 1=通道置换也做成构造式参数族: 指示向量尾部再带 3 个数(π(0),π(1),π(2))/2, "
                         "语义 out[...,c] = in[...,π(c)], 运行时按 π 现造位置轴 gather 表(位置 p 的通道 c=p%3 ⇒ idx[p]=p-c+π(c))。"
                         "⇒ R↔B 互换这类**颜色种类**不再是词表外的东西: 它不是逐通道乘子(MULT_TABLE 那族表达不出置换), "
                         "但它是位置轴上的一个置换, 与 R14 的仿射族共用同一套 gather 机制。"
                         "考试格**不给任何变形名**(名维全零), 只给这 3 个参数 ⇒ 顺带验「规则只靠参数、不靠名」能不能成立。"
                         "0=旧口径(R14 行为逐位不变)")
parser.add_argument("--affine_map", type=int, default=0,
                    help="[R14 种类维参数化] 1=空间算子从'两张写死的表(镜像/roll8,8)'升级为**一般坐标映射**: "
                         "指示向量尾部再带 6 个仿射系数 (a,b,c,d,e,f), 语义 out[Y,X]=in[a·Y+b·X+e, c·Y+d·X+f](取整后 mod IMG), "
                         "运行时按系数**现造 gather 表**。⇒ 竖直翻转/旋转90°/转置 这些都是**同一族的参数取值**, "
                         "不再需要新名槽、新表、新代码、重训。这正是把'种类'从枚举改成参数的做法。"
                         "跑前阳性对照写死: 仿射参数取水平镜像时必须逐位等于 G7 核过的 refl_idx, 取 roll(8,8) 时必须逐位等于 G9 核过的 roll_idx。"
                         "0=旧口径(R13 行为逐位不变)")
parser.add_argument("--exam_unseen_kind", type=int, default=0,
                    help="[R13 尺子] 1=考卷追加**词表外种类**格: 目标是竖直翻转(词表里没有这个名), 而指示只能借用已有的"
                         "'翻转'槽(水平镜像)或什么都不给。量的是最后一维闭集的代价, 并且分清两种失败: "
                         "(a)借槽 ⇒ 模型**自信地执行了错的规则**(水平镜像), (b)无指示 ⇒ 什么都不做。"
                         "(a) 比 (b) 危险: 分数不为零、看着像在工作, 但做的是另一件事。只加考试行, 不改结构/损失/权重")
parser.add_argument("--roll_param", type=int, default=0,
                    help="[R12 构造式几何参数] 1=环形位移量也做成构造式: pval 从 NT 维扩到 NT+2 维, "
                         "第 NT/NT+1 维携带该行的 dY/IMG、dX/IMG; _apply_spat_map 按请求的位移**现造 gather 表**"
                         "(向量化构造, 与 _roll_index 同口径, 跑前逐位核对)。训练只用 (8,8) ⇒ 考试的 (4,4)/(12,12)/(8,0) "
                         "若被覆盖, 就是构造的功劳而不是学来的。需要 --spat_map 1(缺则自动补)。"
                         "约定: 位移两维都为 0 或 pval 只有 NT 维 ⇒ 回落到 (8,8) 那张表(与 R11 逐位相同)")
parser.add_argument("--spat_map", type=int, default=0,
                    help="[R11 构造式空间算子] 1=镜像/平移不再指望网络自己学会路由, 而是**由构造作用在输出的位置轴上**: "
                         "网络只负责内容(未变形的上色结果), 请求了翻转/平移就按已有 gather 表(refl_idx/roll_idx)重排 logits 的位置轴, "
                         "顺序与 apply_combo 一致(先镜像后平移)。与 R9 的 _int_map_M 同一思路(一个作用在档轴、一个作用在位置轴)。"
                         "R10 实测依据: 两条空间支路都开着(‖roll_lin‖=12.65 增益=0.908), 但 G5 锚点判'平移'corr=0.257 远低于执行锚点0.680 "
                         "⇒ 通路在场、规则没被执行; 而颜色两名与翻转都判'执行'。需要 --refl_branch 1(缺则自动补)")
parser.add_argument("--resid_head", type=int, default=0,
                    help="[R30] 1=档内连续残差头: argmax 选档 + 有界残差细化(零初始化, 连续解码=(bin+0.5+0.5*tanh(r))/16), "
                         "loss=CE+resid_loss*L1(连续解码,连续目标); 残差与 logits 同表过位置轴算子。需要连续目标全程保留")
parser.add_argument("--pixel_cont", type=int, default=0,
                    help="[R51] 1=多模态基底上的连续RGB像素头: 接P7联动后的表示q, sigmoid出连续RGB(三原色基), "
                         "loss=像素L1+梯度域L1(像素排列顺序)+同色组一致性。解码用连续头, 不再逐位argmax。非独立图像模型")
parser.add_argument("--pixel_loss", type=float, default=1.0,
                    help="[R51] 像素结构loss权重")
parser.add_argument("--decode_smooth", type=int, default=0,
                    help="[R50] 1=解码级双边滤波去斑(保边): 治逐像素argmax斑点噪声, 确定性无需训练")
parser.add_argument("--refine_cnn", type=int, default=0,
                    help="[R49] 1=卷积精化解码器: bin中心粗解码→3层CNN残差→光滑RGB, L1训练。"
                         "治逐像素独立argmax的斑点噪声(空间不连贯)。末层零初始化⇒初始=粗解码(恒等)")
parser.add_argument("--refine_loss", type=float, default=1.0,
                    help="[R49] 精炼L1损失权重")
parser.add_argument("--soft_ce", type=float, default=0.0,
                    help="[R45] ordinal soft-CE: 目标分布=(1-eps)*onehot+eps/2*邻档(边界归一), 补回 one-hot CE 丢的序数信息; "
                         "0=原 F.cross_entropy 逐位; 解码仍 argmax, 口径不变")
parser.add_argument("--comp_probe", type=int, default=0,
                    help="[R34] 1=考试报探针三角: 源重建acc(规则前logits vs 干净源bins) / copy+规则上限 / 实际acc, "
                         "按行类定位组合偏差产生在哪一环。纯测量, 零训练改动")
parser.add_argument("--int_exact", type=int, default=0,
                    help="[R33] 1=规则构造精确化: P(m) 不查 MIX_GRID 插值, 直接 floor(m*ctr*nb) one-hot"
                         "(与 to_bins 同量化器, 逐位同口径), 去掉 <=0.93 档的结构性插值代价")
parser.add_argument("--resid_input", type=int, default=0,
                    help="[R31] 1=残差头输入追加连续输入值(cat(q, input), 257→1): 档内位置可由输入算出, "
                         "验'q 特征丢档内信息'假设。需要 resid_head=1")
parser.add_argument("--resid_loss", type=float, default=0.5,
                    help="[R30] 残差 L1 项权重(默认0.5)")
parser.add_argument("--train_cross_combo", type=int, default=0,
                    help="[R27] 1=task2 训练行以 30%% 概率换成跨族组合(颜色名×位移/逐通道乘子×位移/通道置换×位移), "
                         "独立随机流 SEED+11 不碰其它流; 目标走 R25 同一个 apply_cross_combo。考试行不动 ⇒ 考卷仍零样本")
parser.add_argument("--exam_unseen_combo", type=int, default=0,
                    help="[R25] 1=追加 3 行族间零样本组合考卷(暖调×位移12 / R↔B×旋转90 / 偏红0.55×位移4), "
                         "各配同目标零指示对照。训练只有单元形+颜色×颜色 ⇒ 跨族组合全是未见。"
                         "目标顺序与 apply_rules 逐字一致: 档轴乘子→通道置换→空间。缺的四族旗标自动补")
parser.add_argument("--exam_unseen_shift", type=int, default=0,
                    help="[R10 尺子] 1=考卷追加**词表外位移量**格: 变形名仍是'平移'(指示向量与训练时逐位相同), "
                         "但目标的环形位移量换成 --unseen_shifts 里的值(训练只见过 (8,8))。"
                         "与 R6 的强度轴同一套对照结构: 输入(灰度)、attr、指示向量全同, 只有目标位移不同 ⇒ "
                         "差值 = 规则在**种类/几何参数**维上覆盖不到时的坍缩量。只加考试行, 不改结构/损失/权重")
parser.add_argument("--unseen_shifts", type=str, default="4,4;12,12;8,0",
                    help="[R10 尺子] 词表外位移量列表, 分号分隔、每项 dY,dX。(8,8) 是训练唯一见过的量, "
                         "故意不放进列表(它是阳性对照, 由 single-平移 那一行充当)")
parser.add_argument("--int_map", type=int, default=0,
                    help="[R9/F2 构造式参数化] 1=档轴算子不再由学习逼近, 而是**由给定的强度 m 直接构造**: "
                         "在他架构已有的冻结候选族 mix_maps=perm_maps(MIX_GRID)(16 点几何网格, 比 2^(-1/6), 覆盖 m∈[0.176,1.0])上, "
                         "按 log2 空间对相邻两格线性插值出 P(m), W=P(m)-I, 再按名连乘。⇒ 词表外的 m 也能出图, "
                         "覆盖由构造给出。R8 已实测'把 m 塞进嵌入模长让网络自己学'会把已见强度一起拖塌(变暗35.01→9.83), 故改走这条。"
                         "需要 mult_compose(连乘路)与 int_cond(强度指示), 缺则自动补 1 并响亮打印。与 mult_oracle/mix/convex 互斥")
parser.add_argument("--mult_compose", type=int, default=0,
                    help="[G12补的一格] 1=给颜色乘子一条**按变形名连乘**的档轴算子: 每通道 c 有 5 张 16×16 表 W[k,c], "
                         "实际算子 M_c = Π_k (I + sel_k·W[k,c])。W 零初始化 ⇒ 第0步 M=I ⇒ logits 逐位不变; "
                         "sel 全 0(去噪/遮补/未请求上色名) ⇒ 对应因子恰为 I ⇒ 那几档逐位不变。"
                         "为什么补它: G11/G12 实测单元形乘子到位(45/49=91.8%)而组合只有 58/119=48.7% 且 k 系统性朝 1 偏, "
                         "现有 prompt_e 是各元件嵌入之**和** ⇒ 身份侧天然只加不乘 ⇒ 推测: 缺的就是连乘这一格")
parser.add_argument("--mult_convex", type=int, default=0,
                    help="[G15关掉负权重逃逸口] 1=乘子族改成**真凸混合**: 去掉 G14 的 tanh 门，"
                         "M_kc = Σ_g softmax(β·(a[k,c] + b·e_0))_g · P(m_g)，β=8.0、b=1.0 都固定不学，"
                         "可学量只剩 a[NT,3,16]=240 个标量。凸 ⇒ M 每个元素天生 ≥0 且行和=1 ⇒ '概率迁移'是构造保证而不是祈祷。"
                         "代价: 起点不再是逐位 I 而是软恒等(w_0=0.995 ⇒ max‖M−I‖≈1.7e-3，自检实测阈值 5e-3)。"
                         "为什么做: G14 实测 4 个不该受影响的通道把 tanh 门学到 −0.87~−0.99 ⇒ M 出负权重，与本臂 CE 7.3206 对 E 臂 4.0531 自洽")
parser.add_argument("--mult_mix", type=int, default=0,
                    help="[G14把参数化收进乘子族] 1=不再有自由 16×16 表, 每个(变形名,通道)只学 16 元候选混合权重 "
                         "a[k,c,g](softmax 温度 β=8.0 固定) 与一个零初始化强度门 gate[k,c]=tanh(·): "
                         "M_kc = I + gate_kc·(Σ_g softmax(β·a)_g·P(m_g) − I)，候选网格 m_g=2^(-g/6)，P(m) 由构造给出、不进梯度。"
                         "gate 与 a 都零初始化 ⇒ 第0步 gate=0 ⇒ M=I **逐位**不变(自检实测)。"
                         "为什么做: G13 证明那张理想图是对的(组合过线 48/49)，G12 证明梯度在自由表上找不到它(行峰度只 0.180) "
                         "⇒ 剩下的唯一候选是参数化形状本身")
parser.add_argument("--mult_oracle", type=int, default=0,
                    help="[G13掰开'表达'与'学习'] 1=bin_gain 不进梯度, 直接填成理想档轴置换图 "
                         "j→to_bins(m_kc·ctr_j) 并 requires_grad=False ⇒ 乘子这一步不给网络学错的机会, 也不给它记忆的空间。"
                         "隐含前提: mult_compose 那条 Π_k(I+sel_k·W_k) 的连乘路必须在(本开关自动把它打开)。"
                         "为什么做: G12 实测交叉项确实存在(0.002550=‖W_a·W_b‖)但组合到位率一位没涨(58/119→58/119)、"
                         "行峰度只有 0.180(理想置换是 1.000)、CE 还涨到 2.8 倍 ⇒ 推测: 缺的是这张图没被学成置换形")
parser.add_argument("--refl_branch", type=int, default=1,
                    help="[G7补的一格] 1=加一条水平反射支路(gather 镜像 + 零初始化 Linear + 有界增益 2·sigmoid)。"
                         "为什么补它: 实测 平移类 2/3 过线、翻转类 0/3 过线, 而引擎现算子集合(shifts=8个±1方向, "
                         "s_seamless=±2平移)只有平移没有镜像 ⇒ 推测:翻转不是读不到提示词而是没有可实现它的通路。"
                         "权重零初始化 ⇒ 第0步与关闭时逐位相同(同 prompt_e/s_seamless 的口径)")
parser.add_argument("--save_png", type=int, default=0,
                    help="[G6验收] 1=考试时把三联图(输入|预测|参考图)落盘; 0=只按原有调用点(save_png=(ep==epochs))")
parser.add_argument("--dump_max", type=int, default=9,
                    help="[G6验收] 三联图最多写几张; 默认 9=旧口径(只覆盖前3张考卷图), 21=全部考卷")
parser.add_argument("--g6_report", type=int, default=1,
                    help="[G6验收] 1=按主人口径打印**与参考图的平均 L1 与皮尔逊相关**(逐任务档/逐图/已训vs未见分层)。"
                         "只读 pred_all/bins 与 exact_audit 同类, 不改损失/结构/权重 ⇒ 不进口径列表")
parser.add_argument("--cell_dev", type=int, default=1,
                    help="[G10查背离] 1=在每格(单元形/组合)那行末尾追加 偏离档数分布(0/1/2/≥3档 + 均值)。"
                         "只读 pred_all 与 bins_flat, 不改损失/结构/权重 ⇒ 与 decode_probe/exact_audit 同类, 不进口径列表。"
                         "用途: 分辨'规则执行了但颜色乘子/量化格子错位'(偏离集中低档) 与'规则没执行'(偏离铺到高档)")
parser.add_argument("--exact_audit", type=int, default=1,
                    help="[G4判据体检] 1=考试时额外打印逐图 pixel_acc 与 exact 命中的图名/任务名。"
                         "只读 pred_all 与 bins_flat, 不改损失/结构/权重 ⇒ 与关闭时同存点分数逐位可比, 故不进口径列表(同 decode_probe 的先例)")
parser.add_argument("--decode_probe", type=int, default=1,
                    help="[尺子诊断] 考试时同时打印三种解码口径: (1)逐位 argmax=现行分数; "
                         "(2)期望值→与 to_bins 同一个量化器取 bin; (3)两种预测各自的 |预测-目标|<=1 档容错率。"
                         "动机(实测): 上色在 5 个臂/两种监督/跑满 400 轮都平在 47~49 ⇒ 需区分"
                         "'模型没学到颜色' 与 '逐 bin argmax 这把尺子把已学到的颜色判错'。"
                         "只读数不改损失不改权重 ⇒ 与历史分数同存点可比, 默认开")
parser.add_argument("--ycbcr_head", type=int, default=0,
                    help="[架构·色度因子化输出头] 1=在同一个位内特征 q 上新增 Y/Cb/Cr 三个 16 阶分类头, "
                         "用**固定精确**的 YCbCr 逆变换把三者的期望值合成期望 RGB, 再经"
                         "'到各 bin 中心的负平方距离'转成 RGB logits 修正项, 以有界可学门 "
                         "G=1+tanh(内容+条件) 加到原逐通道 logits 上。"
                         "为什么是这条路(不是再加损失): color_aux 已从损失侧试过, E200 上色只 +0.26(地板 0.6~0.7); "
                         "缺的是输出字母本身——3 通道各自独立 argmax 要连对 3 次, 而相邻 bin 在数值上几乎同色, "
                         "等权 CE 对此无感。默认 0 ⇒ 结构逐位与现版本相同(新增子模块仍会占参数, 见启动打印的参数量对账)")
parser.add_argument("--ycbcr_aux", type=float, default=0.25,
                    help="[架构·色度因子化输出头] Y/Cb/Cr 三路自身 CE 的合计权重(仅 ycbcr_head=1 时生效)。"
                         "取值不拍: 头吃归一化 q 后初值 CE 每路≈ln16=2.77, 三路合计≈8.3, 而冠军主 CE=1.99 "
                         "⇒ 0.25×8.3=2.1 与主 CE 同阶。每次考试都打印这个比值, 落在 0.2~5 之外就是量级不匹配")
parser.add_argument("--chroma_span", type=float, default=0.20,
                    help="[架构·色度因子化输出头] Cb/Cr 的 16 格铺在 0.5±span 上, 即格宽 span*2/16。"
                         "取值不拍: 冠军存点上实测目标越界比例 "
                         "span0.05→35.3%% 0.10→15.7%% 0.15→10.8%% 0.20→6.8%% 0.25→2.6%% 0.30→1.3%%, "
                         "而格宽 0.20 档 = 0.025 仍是 RGB 格宽(0.0625)的 2.5 倍细 ⇒ 取 0.20 作"
                         "「既不太钳色、又把分辨率真给了色度」的折中; 越界比例每次考试都打印")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--prompt_cond", type=int, default=0,
                    help="[G3正式入口] 1=上色任务的提示词(原色/偏红/偏绿)走 prompt_e embedding, "
                         "同时进条件向量(归一化之后注入)和 P3 attr 侧; 目标图按提示词重着色")

args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

LABELED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img2img_data")
GENERIC_DIR = os.environ.get("M2_GENERIC_DIR", "D:/ai/output/images")   # [跨机] 环境变量可覆盖(远端Linux用)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img2img_out_m2")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(BASE_DIR, "logs", f"{'i2i_eval' if args.exam_only else 'i2i_m2'}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
# [防污染] 看门狗靠 `ls -t i2i_m2_*.txt|head -1` 认当前轮、靠日志里的 [完成] 判跑完 ⇒ 只评测必须换前缀, 否则会把在跑的臂误判成已结束
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
log = open(LOG_PATH, "w", encoding="utf-8")
def w(s): log.write(s + "\n"); log.flush(); print(s, flush=True)

w("=" * 70)
w(f"  V22.1 图生图M2 | SEED={SEED} | bins={args.n_bins} | bs={args.batch_size} | lr={args.lr}")
w(f"  64×64 patch化 + P7跨patch路由 | epochs={args.epochs} use_generic={args.use_generic}")
w(f"  [数据池] generic={GENERIC_DIR} | labeled={LABELED_DIR}")   # [仪器] 换池走环境变量, argv 完全相同 ⇒ 只打日志才能断代(A/C 就靠这行区分)
w("=" * 70)
w("[argv] " + " ".join(sys.argv[1:]))   # [仪器] M2 侧原本没有 argv 回显 ⇒ 事后无法从日志判定跑的哪套开关
if (args.mult_oracle or args.mult_mix or args.mult_convex) and not args.mult_compose:
    # oracle/mix/convex 填的都是 bin_gain 那张表(或它的族内替代参数) ⇒ 没有连乘路它们无处可接; 自动打开而不是报错,
    # 免得各臂因写法不同多一列口径轴
    args.mult_compose = 1
    w(f"[开关自动补] mult_oracle={int(args.mult_oracle)} mult_mix={int(args.mult_mix)} "
      f"mult_convex={int(args.mult_convex)} ⇒ mult_compose 一并置 1")
_nmode = int(args.mult_oracle) + int(args.mult_mix) + int(args.mult_convex)
if _nmode > 1:
    w(f"[配置冲突] mult_oracle/mult_mix/mult_convex 三种参数化互斥，本次给了 {_nmode} 个 ⇒ 选一个"); sys.exit(2)
if args.roll_param and not args.spat_map:
    args.spat_map = 1
    w("[开关自动补] roll_param=1 是 spat_map 的参数化版, 没有构造式空间算子无处可接 ⇒ spat_map 一并置 1")
if args.spat_map and not args.refl_branch:
    args.refl_branch = 1
    w("[开关自动补] spat_map=1 要用 refl_idx/roll_idx 这两张 gather 表, 它们只在 --refl_branch 1 时注册 ⇒ 一并置 1")
# [R25] 跨族组合考卷要四族参数路都在场; 补齐后走下面既有的 chan_mult→int_map→{mult_compose,int_cond,prompt_cond} 链。
if args.exam_unseen_combo:
    _off25 = [k for k in ("chan_mult", "chan_perm", "affine_map", "spat_map") if not getattr(args, k)]
    if _off25:
        for _k25 in _off25:
            setattr(args, _k25, 1)
        w(f"[开关自动补] exam_unseen_combo=1 需要四族参数路: {_off25} 一并置 1")
# [R21] chan_mult 的跑前自检要读 _int_map_M / mix_maps, 而 mix_maps 由 int_map 路初始化。
#   原先这条依赖是隐式的(冠军 argv 恰好带 int_map 才没暴露), 单独开 --chan_mult 会
#   NameError(_NB) 之后又 AttributeError(mix_maps)。按同文件的「开关自动补」风格显式补齐。
if args.chan_mult and not args.int_map:
    args.int_map = 1
    w("[开关自动补] chan_mult=1 的构造自检依赖 _int_map_M/mix_maps ⇒ int_map 一并置 1(原值0)")
if args.int_map:
    if _nmode:
        w("[配置冲突] --int_map 是**构造式**档轴图(不学), 与 mult_oracle/mult_mix/mult_convex 三种学习式参数化互斥 ⇒ 选一个"); sys.exit(2)
    if not (args.mult_compose and args.int_cond):
        _o9 = (int(args.mult_compose), int(args.int_cond))
        args.mult_compose = 1; args.int_cond = 1
        w(f"[开关自动补] int_map=1 要接在连乘路上、且强度必须进指示向量: 原值(mult_compose,int_cond)={_o9} ⇒ 一并置 1")
    # [R23] mix_maps 的注册块在 `if args.prompt_cond and args.mult_compose:` 里 ⇒
    #   只补 mult_compose 不补 prompt_cond 时, 单独开 chan_mult/int_map 仍会 AttributeError(mix_maps)
    #   (§25 未闭那格的根因)。依赖链补全: chan_mult → int_map → {mult_compose, int_cond, prompt_cond}。
    if not args.prompt_cond:
        args.prompt_cond = 1
        w("[开关自动补] int_map=1 的 mix_maps 注册在 prompt_cond&mult_compose 块内 ⇒ prompt_cond 一并置 1(原值0)")
if args.pure_learn_ops and not (args.spatial_upgrade and args.gate_pair):
    w("[配置错误] --pure_learn_ops 1 需要 --spatial_upgrade 1 --gate_pair 1 (PL 改的正是 S 路那 14 个常数加法和 gate_pair)"); sys.exit(2)
if args.pure_learn_ops and args.transform_v2:
    w("[配置错误] --pure_learn_ops 与 --transform_v2 互斥 (v2 走 _forward_v2, 不经过被改的 S 路)"); sys.exit(2)

if args.smoke:
    args.epochs = 3; args.gen_max = 40; args.k_generic = 2
    args.exam_n_generic = 10; args.regen_every = 2; args.exam_every = 1
    w("[SMOKE] 冒烟模式")

# ════════════════════════ 数据 ════════════════════════
IMG = 64          # 整图边长
PATCH = 16        # patch 边长
TABLE_CACHE_CAP = 64    # [R19] 构造式 gather 表缓存上限(每个缓存独立), 防连续参数族无限增表
# [R21] 自检块共享的档数常量原先只在 :1627 的某个 if 分支里定义,
#   单独开 --chan_mult 时 :2570 会 NameError(冠军 argv 因带 --int_map 恰好先初始化才没暴露)。
#   提到模块级无条件定义 ⇒ 各跑前自检不再有隐式先后依赖, 取值与 args.n_bins 恒等。
_NB = args.n_bins
NPATCH = 16       # 4×4
N_IN = PATCH * PATCH * 3          # 768 每 patch
N_POS = NPATCH * N_IN             # 12288 总输出位

def imread_u8(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

def imwrite_u8(path, img_bgr):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img_bgr)
    if ok: buf.tofile(path)

def load_rgb(path):
    img = imread_u8(path)
    if img is None: return None
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
    bgr = img.copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, bgr

def to_patch_major(rgb):
    """[64,64,3] 行主序 → [16,768] patch-major (p = py*4+px, patch 内 16×16×3 行主序)"""
    return rgb.reshape(4, 16, 4, 16, 3).transpose(0, 2, 1, 3, 4).reshape(NPATCH, N_IN)

def from_patch_major(flat):
    """[12288] patch-major → [64,64,3] (PNG 还原用)"""
    return flat.reshape(4, 4, 16, 16, 3).permute(0, 2, 1, 3, 4).reshape(IMG, IMG, 3)   # torch.Tensor.transpose 只收两维参数，原 numpy 风格写法一调用必崩

    def _geo_index(self, pval):
        """[R52] 构造几何位置索引: 把位置值编码进logits, 过已验证的_apply_chan_map/_apply_spat_map
           (纯位置gather), 读回源位置。零重写gather逻辑, 与apply_rules同口径。返回[b,12288]源位置。"""
        b = pval.shape[0]; dev = pval.device
        NP = NPATCH * N_IN
        posval = torch.arange(NP, device=dev, dtype=torch.float32) / NP
        lg = posval.view(1, NP, 1).expand(b, NP, args.n_bins).reshape(b, NPATCH, N_IN, args.n_bins).contiguous()
        lg = self._apply_chan_map(lg, pval)
        lg = self._apply_spat_map(lg, pval)
        return (lg.reshape(b, NP, args.n_bins)[:, :, 0] * NP).round().long().clamp(0, NP - 1)

    def _color_mult(self, pval):
        """[R52] 构造逐通道颜色乘子[b,3]: 名乘子连乘 × chan_mult参数。与_int_map_M同口径(连续版)。"""
        b = pval.shape[0]; dev = pval.device
        mt = torch.tensor(MULT_TABLE, device=dev, dtype=torch.float32)
        x = pval[:, :NT].to(dev)
        eff = torch.where(mt.unsqueeze(0) < 1.0, x.unsqueeze(-1), torch.ones(b, NT, 3, device=dev))
        eff = torch.where((x > 0).unsqueeze(-1), eff, torch.ones_like(eff))
        m = eff.prod(dim=1)
        if args.chan_mult and pval.shape[1] >= PV_W:
            cm = pval[:, NT + 11:NT + 14].to(dev)
            act = cm.abs().sum(1) > 0
            m = torch.where(act.unsqueeze(-1), m * cm, m)
        return m

def _pixel_struct_loss(pred, tc):
    """[R51] 像素级结构loss: 像素L1 + 梯度域L1(像素排列顺序/空间连贯) + 同色组一致性。
       pred,tc [b,12288] patch-major 连续RGB。在多模态表示上直接监督像素结构。"""
    pi = _pm_to_img(pred); ti = _pm_to_img(tc)                 # [b,64,64,3]
    l1 = F.l1_loss(pi, ti)
    pdx = pi[:, :, 1:, :] - pi[:, :, :-1, :]; tdx = ti[:, :, 1:, :] - ti[:, :, :-1, :]
    pdy = pi[:, 1:, :, :] - pi[:, :-1, :, :]; tdy = ti[:, 1:, :, :] - ti[:, :-1, :, :]
    grad = F.l1_loss(pdx, tdx) + F.l1_loss(pdy, tdy)           # 排列顺序: 邻像素差匹配目标
    tq = (ti * 8).long().clamp(0, 7)
    gid = (tq[..., 0] * 64 + tq[..., 1] * 8 + tq[..., 2]).reshape(ti.shape[0], -1)
    pf = pi.reshape(ti.shape[0], -1, 3)
    ngrp = 512
    ones = torch.ones_like(gid, dtype=pi.dtype)
    cnt = torch.zeros(ti.shape[0], ngrp, device=pi.device).scatter_add_(1, gid, ones)
    sm = torch.zeros(ti.shape[0], ngrp, 3, device=pi.device).scatter_add_(
        1, gid.unsqueeze(-1).expand(-1, -1, 3), pf)
    gmean = (sm / cnt.clamp(min=1).unsqueeze(-1)).gather(1, gid.unsqueeze(-1).expand(-1, -1, 3))
    samecolor = (pf - gmean).abs().mean()                      # 同色编号位置: 同目标色组内预测一致
    return l1 + grad + 0.5 * samecolor

def _smooth_img(img):
    """[R50] 双边滤波: 平滑色斑但保留边缘。img [64,64,3] float[0,1] numpy。"""
    u8 = (img * 255).round().astype(np.uint8)
    sm = cv2.bilateralFilter(u8, d=5, sigmaColor=25, sigmaSpace=5)
    return sm.astype(np.float32) / 255.0

def _pm_to_img(flat_b):
    """[R49] [b,12288] patch-major → [b,64,64,3]"""
    b = flat_b.shape[0]
    return flat_b.reshape(b, 4, 4, 16, 16, 3).permute(0, 1, 3, 2, 4, 5).reshape(b, IMG, IMG, 3)

def _img_to_pm(img_b):
    """[R49] [b,64,64,3] → [b,12288] patch-major (to_patch_major 的批量逆)"""
    b = img_b.shape[0]
    return img_b.reshape(b, 4, 16, 4, 16, 3).permute(0, 1, 3, 2, 4, 5).reshape(b, NPATCH * N_IN)

labeled_files = sorted(glob.glob(os.path.join(LABELED_DIR, "*.bmp")))
# [E5修复] 递归 glob: output/images 的图大多在子目录, 非递归只能看到顶层 294 张
generic_files = sorted(sum([glob.glob(os.path.join(GENERIC_DIR, "**", e), recursive=True) for e in ("*.jpg", "*.png", "*.jpeg")], []))
_rng_files = np.random.default_rng(SEED)
generic_files = list(np.array(generic_files)[_rng_files.permutation(len(generic_files))][:args.gen_max + args.exam_n_generic])

sources, _bad = [], 0
for f in labeled_files:
    r = load_rgb(f)
    if r is None: _bad += 1; continue
    sources.append({"name": os.path.basename(f), "kind": "labeled", "rgb": r[0], "bgr": r[1],
                    "rgb_pm": to_patch_major(r[0]).reshape(-1)})
_generic_pool = []
for f in generic_files:
    r = load_rgb(f)
    if r is None: _bad += 1; continue
    _generic_pool.append({"name": os.path.basename(f), "kind": "generic", "rgb": r[0], "bgr": r[1],
                          "rgb_pm": to_patch_major(r[0]).reshape(-1)})

_exam_names = {f"apple_{i:02d}.bmp" for i in range(16, 21)} | {"sun_05.bmp", "apple_01.bmp"}   # [G8 主尺] 训练那张也进考卷：n=7 图 × 7 组合 = 49 格
exam_labeled = [s for s in sources if s["name"] in _exam_names]
train_labeled = [s for s in sources if s["name"] not in _exam_names or s["name"] == "apple_01.bmp"]
exam_generic = _generic_pool[-args.exam_n_generic:]
train_generic = _generic_pool[:-args.exam_n_generic]

w(f"[数据] 带标签(train/exam): {len(train_labeled)}/{len(exam_labeled)} | 通用(train/exam): {len(train_generic)}/{len(exam_generic)} | 坏图: {_bad}")

TASKS = ["去噪", "遮补", "上色"]

# ════════ G5 变形规则: 5 个变形名走同一个 prompt_e 入口(multi-hot), 训练只出单元形, 考试只出未见二元组合 ════════
T_NAMES = ["偏红", "偏绿", "变暗", "翻转", "平移"]
NT = len(T_NAMES)
# 每个变形名对每个通道的乘子(与 apply_combo 同源; 翻转/平移不动颜色 ⇒ 1.0)。只留这一份定义:
# [G12] 天花板自检与 [G13] 的 oracle 档轴图都读它, 免得两处各自硬编码再漂移(G11 探针的轴序错就是这类)。
MULT_TABLE = np.array([[1.0, .55, .55], [.55, 1.0, .55], [.7, .7, .7], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                      dtype=np.float32)
# [G14] 候选乘子网格(固定、不学): m_g = 2^(-g/6), g=0..15 ⇒ 网格比 0.891, 目标 0.55 落在 g=5(0.5612) 与 g=6(0.5) 之间,
#       0.7 落在 g=3(0.7071)。把"能学的东西"限定成"在这族置换图里选一个(再加一个强度门)" ⇒ 参数化形状由构造给定。
MIX_BETA = 8.0                                      # softmax 温度(固定不学)
CONVEX_BIAS = 1.25                                  # [G15] 恒等候选的对数偏置(固定不学): β·b=10 ⇒ 初值 w_0=0.99932
                                                    #   跑前算账: b=1.0 时单名偏差 3.3e-4 看着过 5e-3，但 5 名全请求要连乘 5 次
                                                    #   ⇒ max‖M−I‖ 实测 0.0248(自检当场拦下)。声明的阈值不动、只把手段提到 1.25
                                                    #   (b=4.0 更会让 ∂/∂a 死在 1e-13 ⇒ 也不行)
MIX_GRID = np.power(2.0, -np.arange(16, dtype=np.float32) / 6.0)
EXAM_COMBOS = [(0, 3), (1, 4), (2, 3), (0, 2), (1, 3), (0, 4), (2, 4)]   # 7 个二元组合, 每个元件训练里都见过、成对都没见过
EXAM_ROW_NAMES = []      # [G8] per_task 造考卷时逐行记源图名 ⇒ 追加单元形后"行数≠图数×3", 分组必须按行名而不是按 i//3
PV_W = NT + 14           # [R14/R15/R16] 前 NT 维=名(×强度), NT..NT+1=位移 dY/dX(÷IMG), NT+2..NT+7=仿射 a,b,c,d,e÷IMG,f÷IMG,
                         #   NT+8..NT+10=通道置换 π(0..2)/2, NT+11..NT+13=逐通道乘子 (mR,mG,mB)
                         #   每个参数块"全零=未请求"; 恒等元都是非零的(恒等置换=(0,0.5,1.0)、恒等乘子=(1,1,1)) ⇒ 与未请求可区分
EXAM_ROW_KIND = []
EXAM_SRC_BIN = []       # [R34] 与考卷行对齐的干净源图 bins(仅 comp_probe+per_task 时填)
EXAM_SRC_CONT = []      # [R41] 与考卷行对齐的干净源图连续值(patch-major)       # [R6] 与 EXAM_ROW_NAMES 同长的"这一行考的是什么"标签: base/seen-int/unseen-int<乘子>。
                         #   跨轮比较只按这个标签分组, 不再按行号猜(G11 按 i//8 猜栽过 4 格误配)

def mhot(sel):
    v = np.zeros(NT, dtype=np.float32)
    for i in sel: v[i] = 1.0
    return v

def combo_label(sel):
    _s = "".join(T_NAMES[i] + "+" for i in range(NT) if sel[i] > 0.5)
    return _s[:-1] if _s else "恒等"

def apply_combo(rgb, sel):
    """施加顺序固定 ⇒ 确定性: 颜色(偏红/偏绿/变暗) → 水平翻转 → 环形平移(8,8)"""
    out = rgb.copy()
    if sel[0] > 0.5: out[..., 1] *= 0.55; out[..., 2] *= 0.55
    if sel[1] > 0.5: out[..., 0] *= 0.55; out[..., 2] *= 0.55
    if sel[2] > 0.5: out *= 0.7
    if sel[3] > 0.5: out = out[:, ::-1, :]
    if sel[4] > 0.5: out = np.roll(out, (8, 8), axis=(0, 1))
    return np.clip(out, 0.0, 1.0)

def apply_combo_m(rgb, sel, mvec=None, shift=None):
    """[R6] apply_combo 的乘子可传参版: mvec[NT,3] 给出每个变形名对每个通道的乘子, None ⇒ 用 MULT_TABLE(=写死的 0.55/0.7)。
    存在的唯一理由是造"词表外强度"考卷: 指示向量还是 mhot([k])(词表里没有强度这一位), 目标却按 mvec 生成。
    顺序与 apply_combo 逐字一致(颜色按名序 → 翻转 → 平移), mvec=MULT_TABLE 时两者必须逐位相同(跑前阳性对照见下)。
    [R10] shift=(dY,dX) 非空 ⇒ 环形位移改用这个量(默认 (8,8)=训练唯一见过的量)。同一理由: 词表里没有"位移量"这一位。"""
    mv = MULT_TABLE if mvec is None else np.asarray(mvec, dtype=np.float32)
    sh = (8, 8) if shift is None else tuple(int(v) for v in shift)
    out = rgb.copy()
    if len(sel) > NT and sel[NT] > 0.5:      # [R13] 第 6 位=词表外种类"竖直翻转"; 只考卷用, 训练永不出现
        out = out[::-1, :, :]
    if sel[0] > 0.5: out = out * mv[0]
    if sel[1] > 0.5: out = out * mv[1]
    if sel[2] > 0.5: out = out * mv[2]
    if sel[3] > 0.5: out = out[:, ::-1, :]
    if sel[4] > 0.5: out = np.roll(out, sh, axis=(0, 1))
    return np.clip(out, 0.0, 1.0)

def apply_cross_combo(rgb, sel, mvec, cm, pm, af, sh):
    """[R25] 跨族组合目标: 顺序与模型 apply_rules 逐字一致——档轴乘子(名×强度/逐通道乘子) → 通道置换 → 空间(仿射/位移)。
    每族复用现成单族 numpy 函数 ⇒ 退化闸可逐位核对; 不新造第二套算子(防两处漂移)。"""
    if cm is not None:
        out = apply_chan_mult(rgb, cm)
    elif sel is not None and np.asarray(sel[:3]).max() > 0.5:
        out = apply_combo_m(rgb, sel, mvec, None)
    else:
        out = rgb.copy()
    if pm is not None:
        out = apply_chan_perm(out, pm)
    if af is not None:
        out = apply_affine(out, af)
    elif sh is not None:
        out = np.clip(np.roll(out, tuple(int(v) for v in sh), axis=(0, 1)), 0.0, 1.0)
    return out

def _refl_index():
    """[G7] patch-major 布局下的水平镜像索引表: out[p] = in[refl_idx[p]], refl_idx[p] = 光栅(Y, IMG-1-X) 的 patch-major 下标。
    镜像是对合(翻两次回原样) ⇒ 同一张表既能读也能核。索引口径照 _nbr_table 的写法:
      p = ((Y//16)*4 + X//16)*N_IN + (Y%16)*48 + (X%16)*3 + c"""
    idx = torch.zeros(NPATCH * N_IN, dtype=torch.long)
    for p in range(NPATCH * N_IN):
        patch_i, rem = divmod(p, N_IN)
        y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
        c = rem % 3
        py, px = divmod(patch_i, 4)
        Y, X = py * PATCH + y_in, px * PATCH + x_in
        Xr = IMG - 1 - X
        idx[p] = ((Y // 16) * 4 + Xr // 16) * N_IN + (Y % 16) * 48 + (Xr % 16) * 3 + c
    return idx

def _roll_index(dY=8, dX=8):
    """[G9] patch-major 下的环形平移索引: out[Y,X] = in[(Y-dY)%64, (X-dX)%64] ⇒ 与 np.roll(rgb,(dY,dX),(0,1)) 同一个置换"""
    idx = torch.zeros(NPATCH * N_IN, dtype=torch.long)
    for p in range(NPATCH * N_IN):
        patch_i, rem = divmod(p, N_IN)
        y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
        c = rem % 3
        py, px = divmod(patch_i, 4)
        Y, X = (py * PATCH + y_in - dY) % IMG, (px * PATCH + x_in - dX) % IMG
        idx[p] = ((Y // 16) * 4 + X // 16) * N_IN + (Y % 16) * 48 + (X % 16) * 3 + c
    return idx

_refl_tbl = _refl_index()
_roll_tbl = _roll_index()
with torch.no_grad():
    # [G7 阳性对照·跑前写死] 两条独立实现必须逐位相同: numpy 的 fliplr(经 apply_combo) vs 上面这张 gather 表。
    #   差值不为 0 ⇒ 反射表配错, 当场停下, 后面任何读数都不算(这是仪器, 不是判负闸)。
    _src0 = sources[0]["rgb"]
    _np_path = to_patch_major(apply_combo(_src0, mhot([3]))).reshape(-1)                    # 先镜像再展平
    _gb_path = to_patch_major(_src0).reshape(-1)[_refl_tbl.numpy()]                          # 先展平再按表取
    _rd = float(np.abs(_np_path.astype(np.float64) - _gb_path.astype(np.float64)).max())
    w(f"[G7反射自检] 首张图源 vs 镜像表 最大逐位差={_rd:.6f} (判据: 恒等于 0.000000)")
    # [G9 阳性对照·同口径] 平移表 vs numpy 的 np.roll(经 apply_combo)
    _np9 = to_patch_major(apply_combo(_src0, mhot([4]))).reshape(-1)
    _gb9 = to_patch_major(_src0).reshape(-1)[_roll_tbl.numpy()]
    _rd9 = float(np.abs(_np9.astype(np.float64) - _gb9.astype(np.float64)).max())
    w(f"[G9平移自检] 首张图源 vs roll(8,8) 表 最大逐位差={_rd9:.6f} (判据: 恒等于 0.000000)")
    if _rd != 0.0 or _rd9 != 0.0:
        w("[自检不符] gather 表与 apply_combo 的置换不是同一件事 ⇒ 停在这里, 不开训"); sys.exit(3)
    # [R6 阳性对照·跑前写死] apply_combo_m(mvec=MULT_TABLE) 必须与 apply_combo 逐位相同(5 个单元形 + 7 个考试组合),
    #   否则"词表外强度"那格的差值里会混进两条实现路径的差, 读数就不是词表的锅了。
    _r6bad = []
    for _s6 in [mhot([k]) for k in range(NT)] + [mhot(c) for c in EXAM_COMBOS]:
        _a6 = apply_combo(_src0, _s6).astype(np.float64)
        _b6 = apply_combo_m(_src0, _s6, MULT_TABLE).astype(np.float64)
        if float(np.abs(_a6 - _b6).max()) != 0.0: _r6bad.append(combo_label(_s6))
    # 仪器有效性: 词表外乘子必须真的改动目标(逐位差>0), 否则那一格是空考, 分数不动说明不了任何事
    _r6mv = MULT_TABLE.copy(); _r6mv[0] = [1.0, 0.40, 0.40]
    _r6d = float(np.abs(apply_combo(_src0, mhot([0])).astype(np.float64)
                        - apply_combo_m(_src0, mhot([0]), _r6mv).astype(np.float64)).max())
    w(f"[R6强度自检] apply_combo_m(MULT_TABLE) vs apply_combo 不符项={_r6bad if _r6bad else '无(12/12 逐位相同)'}"
      f" | 偏红 m 0.55→0.40 目标最大逐位差={_r6d:.6f} (判据: >0, 否则空考)")
    if _r6bad or _r6d <= 0.0:
        w("[R6自检不符] 参数化乘子版与原实现不同源, 或词表外乘子没改动目标 ⇒ 停在这里, 不开训"); sys.exit(3)

SIGMA_POOL = [0.10, 0.20, 0.30]   # E4: 去噪课程当前档(全局; 考试集构建时为全档)
def set_sigma_pool(ep):
    """E4 课程: 前1/3轮只学σ0.1, 中1/3加σ0.2, 后1/3全档 (E4实测负优化, 默认关)"""
    if not args.sigma_curriculum:
        SIGMA_POOL[:] = [0.10, 0.20, 0.30]; return
    e3 = max(1, args.epochs // 3)
    if ep <= e3: SIGMA_POOL[:] = [0.10]
    elif ep <= 2 * e3: SIGMA_POOL[:] = [0.10, 0.20]
    else: SIGMA_POOL[:] = [0.10, 0.20, 0.30]

def degrade(rgb, task, rng):
    """返回 (退化图, task, sigma, mask摘要[3]); mask摘要只标"已知合成坏区"(遮补块), 去噪/上色无已知坏区=0"""
    out = rgb.copy()
    sigma = 0.0
    mfeat = np.zeros(3, dtype=np.float32)
    if task == 0:
        sigma = float(rng.choice(SIGMA_POOL))
        out = np.clip(rgb + rng.normal(0.0, sigma, rgb.shape).astype(np.float32), 0.0, 1.0)
    elif task == 1:
        k = int(rng.choice([16, 24, 32]))          # 64×64: 遮块同比放大
        y = int(rng.integers(0, IMG - k + 1)); x = int(rng.integers(0, IMG - k + 1))
        out[y:y + k, x:x + k] = 0.5
        mfeat[:] = [k * k / (IMG * IMG), (y + k / 2) / IMG, (x + k / 2) / IMG]
    else:
        g = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float32)
        out = np.stack([g, g, g], axis=-1)
    return np.clip(out, 0.0, 1.0), task, sigma, mfeat

def to_bins(rgb_flat):
    return np.clip((rgb_flat * args.n_bins).astype(np.int64), 0, args.n_bins - 1)

def perm_maps(mvals, nb):
    """[G,nb,nb] 理想档轴置换图: P(m) 把输入档 j 送到 to_bins(m·ctr_j)。G13 的 oracle 与 G14 的候选族共用这一处构造
       (只留一份实现: G12 那批读数栽过"两处各自硬编码再漂移"的坑)。"""
    ctr = (np.arange(nb, dtype=np.float32) + 0.5) / nb
    out = np.zeros((len(mvals), nb, nb), dtype=np.float32)
    for gi, m in enumerate(np.asarray(mvals, dtype=np.float32).reshape(-1)):
        out[gi][np.arange(nb), to_bins(m * ctr).astype(np.int64)] = 1.0
    return torch.from_numpy(out)

def oracle_bin_maps(nb):
    """[G13] 理想档轴迁移表 W[k,c,j,m] = P(m_{k,c})[j,m] − I[j,m]。
    填进 bin_gain 后，单名时 M=I+(P−I)=P 恰是该乘子的置换图，双名时 M=P_a·P_b=先按 a 量化再按 b 量化(连乘的离散形式)。"""
    P = perm_maps(MULT_TABLE.reshape(-1), nb).view(NT, 3, nb, nb)
    return P - torch.eye(nb, dtype=np.float32)

_p3vis = P3Vis(chroma=bool(args.p3v_chroma), kmeans_seed=SEED)
w(f"[P3Vis] chroma={int(args.p3v_chroma)} | 机理={'色度分布+白平衡增益补进 V1:44-47/V3:34-47' if args.p3v_chroma else '关闭(与旧口径逐位相同)'} | 纹理后端={_p3vis.tex_backend}")
w(f"[P3Vis] kmeans_seed={SEED} ⇒ V1:28-43 这 16 个调色板槽改为逐次可复现"
  f"（旧版此处未播种: 同一张图每次调用都重抽, 实测最大差 0.44, 连同一 run 内部都不自洽）")
if _p3vis.tex_backend != "skimage" and not args.p3v_allow_no_skimage:
    # V3 的 0-23 依赖 skimage; 缺席就静默全零 ⇒ 纹理层在场与否变成机器相关的隐藏口径轴, 响亮拒绝而不是硬跑
    w("[环境不足] 本机没有 scikit-image: P3Vis 纹理层 V3:0-23 会整段恒零(旧代码就是这样静默跑的)。"
      "请先 pip install scikit-image, 或明确加 --p3v_allow_no_skimage 1 承认这一口径"); sys.exit(3)

def apply_affine(rgb, af):
    """[R14] 一般坐标映射的 **numpy 独立实现**(与模型侧 _affine_tbl_for 是两套代码, 专供跑前逐位核对)。
       语义: out[Y,X] = in[round(a·Y+b·X+e) % H, round(c·Y+d·X+f) % W]。
       对照: 水平镜像=(1,0,0,-1,0,W-1)、竖直翻转=(-1,0,0,1,H-1,0)、rot90=(0,1,-1,0,0,W-1)、roll(dY,dX)=(1,0,0,1,-dY,-dX)。"""
    a, b, c, d, e, f = (float(v) for v in af)
    H, W = rgb.shape[:2]
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Ys = np.rint(a * Y + b * X + e).astype(np.int64) % H
    Xs = np.rint(c * Y + d * X + f).astype(np.int64) % W
    return rgb[Ys, Xs]

def apply_chan_mult(rgb, cm):
    """[R16] 逐通道乘子的 **numpy 独立实现**(与模型侧 _Pm 走的档轴插值是两套东西, 专供跑前核对目标)。
       语义: out[...,c] = clip(in[...,c] * cm[c])。对照: 偏红=(1,0.55,0.55)、偏绿=(0.55,1,0.55)、变暗=(0.7,0.7,0.7)。"""
    return np.clip(rgb * np.asarray(cm, dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)

def apply_chan_perm(rgb, perm):
    """[R15] 通道置换的 **numpy 独立实现**(与模型侧 _chan_tbl_for 是两套代码, 专供跑前逐位核对)。
       语义: out[...,c] = in[...,perm[c]]。对照: 恒等=(0,1,2)、R↔B=(2,1,0)、轮转=(1,2,0)。"""
    return rgb[..., list(int(v) for v in perm)]

def _mvec_for(k, m):
    """[R6] 把第 k 个变形名的乘子换成 m, 其余名保持 MULT_TABLE 原值。哪个通道留 1.0 照抄 MULT_TABLE 的形状
       (偏红动 G/B、偏绿动 R/B、变暗三通道同乘) ⇒ 只改强度, 不改'这个名是什么意思'。"""
    mv = MULT_TABLE.copy()
    if k == 0: mv[0] = [1.0, m, m]
    elif k == 1: mv[1] = [m, 1.0, m]
    elif k == 2: mv[2] = [m, m, m]
    return mv

# [R8] 强度抽样用**独立**生成器: 不碰 rng/rng_train 的随机流 ⇒ 非上色行的输入/目标与 B2 逐位相同(单变量)。
#   模块级(不是 build_pairs 内部)⇒ --regen_every 1 每次重建都接着抽, 同一个名能拿到多个不同 m。
_RNG_INT = np.random.default_rng(SEED + 7)
# [R27] 训练侧跨族组合用独立流: 不碰 rng/rng_train/_RNG_INT ⇒ 非组合行输入/目标逐位不变。
_RNG_XCOM = np.random.default_rng(SEED + 11)

def _int_of(mv, k):
    """[R8] 从乘子表反取第 k 个名的强度标量: 颜色名的表行形如 [1,m,m]/[m,1,m]/[m,m,m] ⇒ min 就是 m。
       只读 MULT_TABLE 这一份定义, 不再硬编码第二个 0.55(G12 那批读数栽过两处各自硬编码再漂移的坑)。"""
    return float(np.asarray(mv, dtype=np.float32)[k].min())

def build_pairs(srcs, k_per_img, rng, per_task=False):
    """退化对; attr 双路: 整图64×64 P3Vis(全局h) + 每 patch 16×16 P3Vis(E2: 进P7路由)"""
    list_rgb, list_dgb, list_bin, list_attr_p, list_attr_i, list_task, list_sigma, list_mfeat, list_prompt = \
        [], [], [], [], [], [], [], [], []
    list_pval = []
    list_tc = []                                  # [R30] 连续目标(量化前), 与 list_bin 逐行对齐                              # [R8] 送进模型的指示向量(名×强度); int_cond=0 时与 list_prompt 逐位相同
    t0 = time.time()
    _ci = 0                                     # G5: 考试按考卷图序固定分配未见组合
    _nc27 = 0                                     # [R27] 本次重建产出的跨族组合训练行计数
    _rr = [0]                                   # [R7] 训练侧变形名轮转计数(只在 --train_vocab_cover 1 时被读)
    _umults = [float(x) for x in str(args.unseen_mults).split(",") if x.strip()]   # [R6] 词表外强度列表
    _ilo, _ihi = [float(x) for x in str(args.train_int_range).split(",")][:2]      # [R8] 训练侧强度抽样区间
    # [R10] 词表外位移量列表: "4,4;12,12;8,0" → [(4,4),(12,12),(8,0)]
    _ushifts = [tuple(int(v) for v in _it.split(",")) for _it in str(args.unseen_shifts).split(";")
                if _it.strip() and len(_it.split(",")) == 2]
    if per_task: EXAM_ROW_NAMES.clear(); EXAM_ROW_KIND.clear(); EXAM_SRC_BIN.clear(); EXAM_SRC_CONT.clear()   # [G8/R6] 行名与行类标签都与考卷行数一一对应
    for s in srcs:
        tasks = range(3) if per_task else [int(rng.integers(0, 3)) for _ in range(k_per_img)]
        # [G8] 追加 5 个单元形对照格; 上色退化与单元形都不消耗随机数 ⇒ 原有各行输入/目标逐位不变
        _plan = [(t, None, None, "base", None, None, None, None) for t in tasks]
        if per_task and args.exam_combo_all and args.prompt_cond:
            # [G12 尺子] 每张考卷图配齐 7 个组合 ⇒ 每个组合的 n 从 1 张图升到 7 张图
            #   基础那行的上色格改吃 COMBOS[0](不再按图序轮转), 其余 6 个追加在后面 ⇒ 无重复格
            _plan += [(2, mhot(c), None, "seen-int-combo", None, None, None, None) for c in EXAM_COMBOS[1:]]
        if per_task and args.exam_singles and args.prompt_cond:
            # [R6] 行类标签带上变形名: 它才是 unseen-int 那格的**同口径对照**(同名、训练用过的乘子)
            _plan += [(2, mhot([k]), None, f"single-{T_NAMES[k]}", None, None, None, None) for k in range(NT)]
        if per_task and args.exam_unseen_int and args.prompt_cond:
            # [R6 尺子] 词表外强度格: 指示向量仍是 mhot([k])(与训练时逐位相同, 词表没有强度这一位),
            #   目标却按 m 生成 ⇒ 这一格分数与同名单元形(single-*)之差 = 词表外坍缩量。
            #   只考 3 个颜色名: 翻转/平移是置换, 没有强度可改(改了就变成另一个置换, 属于"未见变形名"另开一格)。
            #   task=2 的 degrade 不消耗随机数 ⇒ 原 21 行输入/目标逐位不变, base 分数仍与历史同口径。
            for _m in _umults:
                for _k in (0, 1, 2):
                    _plan.append((2, mhot([_k]), _mvec_for(_k, _m), f"unseen{_m:.2f}-{T_NAMES[_k]}", None, None, None, None))
        if per_task and args.exam_unseen_shift and args.prompt_cond:
            # [R10 尺子] 词表外位移量格: 名仍是"平移"(mhot([4])), 目标按 (dY,dX) 生成。
            #   输入/attr/指示与 single-平移 那一行逐位相同(task=2 的 degrade 是确定性灰度) ⇒ 差值只由位移量不可表达造成。
            for _sh in _ushifts:
                _plan.append((2, mhot([4]), None, f"unseenShift{_sh[0]}x{_sh[1]}-{T_NAMES[4]}", _sh, None, None, None))
        if per_task and args.exam_unseen_kind and args.prompt_cond:
            # [R13 尺子] 词表外种类: 目标=竖直翻转(词表里没有), 指示分两格——
            #   kind-borrow: 借"翻转"槽(水平镜像) ⇒ 看模型是否自信地执行错的规则
            #   kind-none  : 指示全零 ⇒ 看模型是否什么都不做
            #   两格的目标逐位相同, 只有指示不同 ⇒ 差值 = "闭集词表把一个未知名硬塞进最近的名"的代价
            _vf = np.zeros(NT + 1, dtype=np.float32); _vf[NT] = 1.0
            _br = np.zeros(NT + 1, dtype=np.float32); _br[3] = 1.0; _br[NT] = 1.0
            _plan.append((2, _vf, None, "unseenKind-竖直翻转-无指示", None, None, None, None))
            _plan.append((2, _br, None, "unseenKind-竖直翻转-借翻转槽", None, None, None, None))
        if per_task and args.affine_map and args.prompt_cond:
            # [R14 尺子] 词表外种类**带仿射参数**: 指示仍然只借"翻转"那一个名槽(名维还是闭的),
            #   但几何由 6 个系数给出 ⇒ 竖直翻转/旋转90° 都不需要新名。目标由 numpy 的 apply_affine 独立生成。
            #   与 R13 的"借翻转槽(无参数)"同目标、同指示名, 只差那 6 个系数 ⇒ 差值就是"种类参数化"买到的东西。
            _AFS = [("竖直翻转", (-1.0, 0.0, 0.0, 1.0, float(IMG - 1), 0.0)),
                    ("旋转90", (0.0, 1.0, -1.0, 0.0, 0.0, float(IMG - 1)))]
            for _nm14, _af14 in _AFS:
                _plan.append((2, mhot([3]), None, f"unseenKindAffine-{_nm14}", None, _af14, None, None))
        if per_task and args.chan_perm and args.prompt_cond:
            # [R15 尺子] 颜色种类(通道置换): 名维**全零**, 只给 3 个参数; 目标由 numpy 的 apply_chan_perm 独立生成。
            #   配一个"同目标、零指示"的对照格 ⇒ 差值 = 参数化规则买到的东西(而不是内容本身碰巧对)。
            for _nm15, _pm15 in (("R↔B", (2, 1, 0)), ("轮转RGB", (1, 2, 0))):
                _plan.append((2, np.zeros(NT, dtype=np.float32), None, f"unseenChanPerm-{_nm15}", None, None, _pm15, None))
                # [R15修正] 对照格也带 _pm15 ⇒ **目标与实验格逐位相同**(都是置换后的图), 只是参数不进指示向量。
                #   上一版这里传 None ⇒ 对照格的目标退化成干净图, 而 pixel_acc 对"预测与目标同时被同一置换重排"本就
                #   不变 ⇒ 两格必然逐位相同(实测都是 51.37%), 那个 Δ=+0.00pp 是恒等式不是测量, 不能当证据。
                _plan.append((2, np.zeros(NT, dtype=np.float32), None, f"unseenChanPerm-{_nm15}-零指示对照", None, None, _pm15, None))
        if per_task and args.chan_mult and args.prompt_cond:
            # [R16 尺子] 词表外**颜色规则**: 逐通道乘子三元组, 任何一个都不等于 偏红/偏绿/变暗 的 (1,.55,.55)/(.55,1,.55)/(.7,.7,.7)。
            #   实验格名维全零、只给 3 个参数 ⇒ 与 R15 同一问法: 规则只靠参数能不能成立。
            #   每个配两格对照, 三格**目标逐位相同**(都带 _cm16), 只有指示不同:
            #     零指示对照 = 什么都不给;  借变暗名对照 = 给"变暗"这个名(模型会执行 ×0.7 全通道, 是错的规则)
            _CMS = [("暖调", (1.00, 0.85, 0.70)), ("冷调", (0.80, 0.90, 1.00)), ("压绿", (0.90, 0.60, 0.90))]
            for _nm16, _cm16 in _CMS:
                _z16 = np.zeros(NT, dtype=np.float32)
                _plan.append((2, _z16, None, f"unseenChanMult-{_nm16}", None, None, None, _cm16))
                _plan.append((2, _z16, None, f"unseenChanMult-{_nm16}-零指示对照", None, None, None, _cm16))
                _plan.append((2, mhot([2]), None, f"unseenChanMult-{_nm16}-借变暗名对照", None, None, None, _cm16))
        if per_task and args.exam_unseen_combo and args.prompt_cond:
            # [R25 尺子] 族间零样本组合: 训练只有单元形+颜色×颜色(G12 COMBOS), 这三行全是未见复合。
            #   每行配"同目标零指示"对照: 参数照带(造目标用)但不进 pval, 名维全零 ⇒ Δ=复合规则买到的东西。
            _z25 = np.zeros(NT, dtype=np.float32)
            _X25 = [
                ("暖调x位移12", _z25, None, (1.00, 0.85, 0.70), None, None, (12, 12)),
                ("R换B x旋转90", _z25, None, None, (2, 1, 0),
                 (0.0, 1.0, -1.0, 0.0, 0.0, float(IMG - 1)), None),
                ("偏红0.55x位移4", mhot([0]), _mvec_for(0, 0.55), None, None, None, (4, 4)),
            ]
            for _nm25, _sel25, _mv25, _cm25, _pm25, _af25, _sh25 in _X25:
                _plan.append((2, _sel25, _mv25, f"unseenCross-{_nm25}", _sh25, _af25, _pm25, _cm25))
                _plan.append((2, _z25, None, f"unseenCross-{_nm25}-零指示对照", _sh25, _af25, _pm25, _cm25))
        # [仪器·防呆闸] 同一张图的考卷行元组必须同宽。R16 就栽在这: 新加一格(逐通道乘子)时忘了给旧的
        #   chanPerm 行补位 ⇒ 只报一句 "not enough values to unpack (expected 8, got 7)", 既不说是哪一行、
        #   也不说是哪一格, 白花一轮。宽度不齐当场点名并停, 不进训练。
        _wl = {len(_e) for _e in _plan}
        if len(_wl) != 1:
            _wx = max(_wl)
            w(f"[plan宽度不符] 本图考卷行元组宽度={sorted(_wl)} ⇒ 宽度不足的格子: "
              + ", ".join(str(_e[3]) for _e in _plan if len(_e) != _wx)
              + " (新加一种考卷行时, 必须给所有旧行补位到同宽)"); sys.exit(3)
        for task, _ovr, _mv, _kind, _sh, _af14, _pm15, _cm16 in _plan:
            deg, task, sig, mfeat = degrade(s["rgb"], task, rng)
            _sel = _ovr
            if _sel is None and args.prompt_cond and task == 2:
                # G3 修的是"别借道 mask_h"; G5 把内容从"3 个色档"扩成"5 个变形名 × 组合"
                if per_task:
                    if args.exam_combo_all:
                        _sel = mhot(EXAM_COMBOS[0])            # G12: 基础行固定用第 1 个组合, 其余 6 个由 _plan 追加
                    else:
                        _sel = mhot(EXAM_COMBOS[_ci % len(EXAM_COMBOS)]); _ci += 1
                else:
                    _draw = int(rng.integers(0, NT))                   # 照抽以对齐随机流(非上色行输入/目标逐位不变)
                    if args.train_vocab_cover:
                        _draw = _rr[0] % NT; _rr[0] += 1               # [R7] 按名序轮转 ⇒ 5/5 名必被覆盖
                    _sel = mhot([_draw])                               # 训练只出单元形 ⇒ 组合全是未见
                    if args.int_cond and _draw < 3:
                        # [R8] 训练侧强度随机化: 目标与指示共用同一个 m ⇒ "强度"这一位第一次携带信息、第一次有梯度。
                        #   用 _RNG_INT(独立流) ⇒ 去噪/遮补行与几何名行的输入/目标仍与 B2 逐位相同。
                        _mv = _mvec_for(_draw, float(_RNG_INT.uniform(_ilo, _ihi)))
                    # [R27] 训练侧跨族覆盖: 30% 概率把单元形换成跨族组合。参数连续抽样 ⇒
                    #   与考试固定值(暖调三元组/位移12x12/4x4/偏红0.55)几乎必然不撞, 考卷仍零样本。
                    if args.train_cross_combo:
                        if float(_RNG_XCOM.random()) < 0.3:
                            _fam27 = int(_RNG_XCOM.integers(0, 3))
                            _sh = (int(_RNG_XCOM.integers(2, 15)), int(_RNG_XCOM.integers(2, 15)))
                            if _fam27 == 0:
                                _k27 = int(_RNG_XCOM.integers(0, 3))
                                _sel = mhot([_k27])
                                _mv = _mvec_for(_k27, float(_RNG_XCOM.uniform(_ilo, _ihi)))
                            elif _fam27 == 1:
                                _sel = np.zeros(NT, dtype=np.float32)
                                _cm16 = tuple(float(_RNG_XCOM.uniform(0.5, 1.0)) for _ in range(3))
                            else:
                                _sel = np.zeros(NT, dtype=np.float32)
                                _pm15 = ((2, 1, 0), (1, 2, 0), (1, 0, 2), (2, 0, 1), (0, 2, 1))[int(_RNG_XCOM.integers(0, 5))]
                            _kind = "trainCross"
                            _nc27 += 1
            deg_bgr = (deg[:, :, ::-1] * 255.0).astype(np.uint8)
            attr_i = np.nan_to_num(_p3vis.process_image(deg_bgr).astype(np.float32),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            attr_p = np.empty((NPATCH, 256), dtype=np.float32)     # E2: 16 个 patch 各自属性
            for pi in range(NPATCH):
                py, px = divmod(pi, 4)
                pb = deg_bgr[py * PATCH:(py + 1) * PATCH, px * PATCH:(px + 1) * PATCH]
                attr_p[pi] = np.nan_to_num(_p3vis.process_image(pb).astype(np.float32),
                                           nan=0.0, posinf=0.0, neginf=0.0)
            deg_pm = to_patch_major(deg).reshape(-1)               # [12288] patch-major
            list_rgb.append(deg_pm)
            list_dgb.append(to_bins(deg_pm))
            if _sel is not None:
                # [R15] 通道置换格的名维是全零 multi-hot(不是 None) ⇒ 仍走这条分支, 目标由 _pm15 决定
                # [R6] _mv 非空 ⇒ 目标是"词表外强度"版本; _mv=None ⇒ apply_combo_m 走 MULT_TABLE, 与 apply_combo 逐位相同(跑前自检)
                # [R14] _af14 非空 ⇒ 目标由仿射系数用 numpy 独立生成(与指示解耦: 指示里那个借来的"翻转"名不参与造目标)
                if _kind.startswith(("unseenCross", "trainCross")):
                    _tgt14 = apply_cross_combo(s["rgb"], _sel, _mv, _cm16, _pm15, _af14, _sh)  # [R25] 多族同填, 同序复合
                elif _cm16 is not None:
                    _tgt14 = apply_chan_mult(s["rgb"], _cm16)      # [R16] 逐通道乘子: 与指示解耦, numpy 独立生成
                elif _pm15 is not None:
                    _tgt14 = apply_chan_perm(s["rgb"], _pm15)      # [R15] 颜色种类: 与指示解耦, numpy 独立生成
                elif _af14 is not None:
                    _tgt14 = apply_affine(s["rgb"], _af14)         # [R14] 空间种类: 同上
                else:
                    _tgt14 = apply_combo_m(s["rgb"], _sel, _mv, _sh)
                _pm30 = to_patch_major(_tgt14).reshape(-1)
                list_tc.append(_pm30)                              # [R30] 连续目标保留到 run_batch
                list_bin.append(to_bins(_pm30))                    # [R13] _sel 宽 NT+1 时含竖直翻转位
            else:
                list_tc.append(s["rgb_pm"])                        # [R30]
                list_bin.append(to_bins(s["rgb_pm"]))              # 目标图: 预转换的 patch-major
            list_attr_p.append(attr_p)
            list_attr_i.append(attr_i)
            list_task.append(task)
            list_sigma.append(sig)                                 # E4: 去噪对的σ档位(非去噪=0)
            list_mfeat.append(mfeat)                               # T1: mask摘要(遮补块面积/质心, 其余=0)
            # G5: 5 维 multi-hot。[R13] 词表外种类的 _sel 宽 NT+1(第6位只给 apply_combo_m 造目标用),
            #   这里一律裁回 NT 维 ⇒ 所有按 multi-hot 语义读 prompts 的报告/锚点不受影响
            list_prompt.append(_sel[:NT].copy() if _sel is not None else np.zeros(NT, dtype=np.float32))
            # [R8] 指示向量(送进模型的那一份): int_cond=1 ⇒ 把激活的颜色名那一维换成该行实际乘子, 几何名保持 1.0;
            #   int_cond=0 ⇒ 与 list_prompt 逐位相同(旧口径)。list_prompt 本身不动 ⇒ 所有按 multi-hot 语义读它的
            #   报告/锚点(G5 的 apply_combo、combo_label、prompts[:,3]>0.5)一律不受影响。
            _pv8 = np.zeros(PV_W, dtype=np.float32)            # [R12/R14] 尾部 = dY/IMG, dX/IMG, 仿射 a,b,c,d,e/IMG,f/IMG
            _pv8[:NT] = list_prompt[-1][:NT]                   # [R13] 第6位(词表外种类)没有槽位 ⇒ 进不了指示向量
            if args.int_cond and _sel is not None:
                _mv8 = MULT_TABLE if _mv is None else _mv
                for _k8 in range(NT):
                    if _pv8[_k8] > 0.5: _pv8[_k8] = _int_of(_mv8, _k8) if _k8 < 3 else 1.0
            _is_ctl = "对照" in _kind        # [R25] 提前: 位移/仿射参数块也要对对照格抑制(原先只抑 pm/cm)
            if ((_sel is not None and len(_sel) > 4 and _sel[4] > 0.5) or _sh is not None) and not _is_ctl:
                _sh8 = (8, 8) if _sh is None else (int(_sh[0]), int(_sh[1]))
                _pv8[NT] = _sh8[0] / IMG; _pv8[NT + 1] = _sh8[1] / IMG
            if _af14 is not None and not _is_ctl:
                # 线格式: a~d 原样, e/f 以 ÷IMG 存(与位移同口径); 解码回像素单位在 _apply_spat_map
                _pv8[NT + 2:NT + 6] = np.asarray(_af14[:4], dtype=np.float32)
                _pv8[NT + 6] = float(_af14[4]) / IMG; _pv8[NT + 7] = float(_af14[5]) / IMG
            # [R15修正] 名字里带"对照"的格: 目标照旧(与实验格同一张图), 但**参数块不写进指示向量**
            #   ⇒ 它才是真对照(同一道题、不给参数)。名维照常写, 所以"借某个名"的对照格仍然带着那个名。
            _is_ctl = "对照" in _kind
            if _pm15 is not None and not _is_ctl:
                _pv8[NT + 8:NT + 11] = np.asarray(_pm15, dtype=np.float32) / 2.0   # [R15] π/2, 全零=未请求
            if _cm16 is not None and not _is_ctl:
                _pv8[NT + 11:NT + 14] = np.asarray(_cm16, dtype=np.float32)        # [R16] 逐通道乘子原值, 全零=未请求
            list_pval.append(_pv8)
            if per_task:
                EXAM_ROW_NAMES.append(s["name"]); EXAM_ROW_KIND.append(_kind)
                if args.comp_probe: EXAM_SRC_BIN.append(to_bins(s["rgb_pm"]))   # [R34]
                if args.comp_probe: EXAM_SRC_CONT.append(s["rgb_pm"])            # [R41]          # [G8/R6] 行名与行类对齐
    if not per_task and args.train_cross_combo:
        w(f"[R27 训练组合] 本次重建产出 {_nc27} 行跨族组合训练行(30%概率, 独立流SEED+11)")
    rgb_t = torch.tensor(np.stack(list_rgb), dtype=torch.float32, device=DEVICE)
    dgb_t = torch.tensor(np.stack(list_dgb), dtype=torch.long, device=DEVICE).reshape(len(list_rgb), NPATCH, N_IN)
    bin_t = torch.tensor(np.stack(list_bin), dtype=torch.long, device=DEVICE).reshape(len(list_rgb), NPATCH, N_IN)
    attrp_t = torch.tensor(np.stack(list_attr_p), dtype=torch.float32, device=DEVICE)   # [N,16,256]
    attri_t = torch.tensor(np.stack(list_attr_i), dtype=torch.float32, device=DEVICE)   # [N,256]
    task_t = torch.tensor(list_task, dtype=torch.long, device=DEVICE)
    sigma_t = torch.tensor(list_sigma, dtype=torch.float32, device=DEVICE)
    mfeat_t = torch.tensor(np.stack(list_mfeat), dtype=torch.float32, device=DEVICE)   # [N,3]
    prompt_t = torch.tensor(np.stack(list_prompt), dtype=torch.float32, device=DEVICE)  # [N,5] G5 multi-hot
    pval_t = torch.tensor(np.stack(list_pval), dtype=torch.float32, device=DEVICE)
    tc_t = torch.tensor(np.stack(list_tc), dtype=torch.float32, device=DEVICE)   # [R30] [N,12288] 连续目标      # [N,5] [R8] 名×强度(int_cond=0 时==prompt_t)
    w(f"  [造对] {len(srcs)}图×{3 if per_task else k_per_img} = {len(list_rgb)}对(64×64) | per-patch P3Vis+量化 {time.time()-t0:.1f}s | σ池={sorted(set(SIGMA_POOL))}")
    if not per_task:
        # [R6 词表采样审计] 训练集的变形名是在造对时一次性抽定、之后每轮复用同一批 ⇒ 抽到几次就是学几次,
        #   没抽到的名对应的 prompt_e 行永远拿不到梯度(实测 ‖prompt_e‖ 平移=0.0)。这一行把"词表覆盖"从推断变成计数。
        _hist = {t: 0 for t in T_NAMES}
        for _p in list_prompt:
            for _ki in range(NT):
                if _p[_ki] > 0.5: _hist[T_NAMES[_ki]] += 1
        _nz = sum(1 for v in _hist.values() if v > 0)
        w(f"  [词表采样] 训练集变形名计数={_hist} | 任务计数={dict(zip(TASKS, [int((np.array(list_task) == t).sum()) for t in range(3)]))}"
          f" | 覆盖 {_nz}/{NT} 名" + ("" if _nz == NT else f" ⇒ 缺 {NT-_nz} 名一次都没训到, 该名的指示位在考试里必然是死的"))
        if args.int_cond:
            # [R8 强度审计] 训练里出现过的 m 值。若只有 1~2 个不同值, 那"学会读 m"与"背下这几个常数"就分不开,
            #   考试上的收窄不能当作 F2 成立的证据 ⇒ 这一行是判读坍缩量之前的前置检查。
            _ms = sorted({round(float(v[_ki]), 4) for v in list_pval for _ki in range(3) if v[_ki] > 0.5})
            w(f"  [强度采样] 训练用过的 m={_ms} | 不同值 {len(_ms)} 个 | 区间={args.train_int_range}"
              + (" ⇒ 只有一个值, 学不出'读 m'(需要 --regen_every 1 让每次重建重抽)" if len(_ms) <= 1 else ""))
    return rgb_t, dgb_t, bin_t, attrp_t, attri_t, task_t, sigma_t, mfeat_t, prompt_t, pval_t, tc_t

# ════════════════════════ 模型 ════════════════════════
def _orth_init(t, gain=1.0):
    """[跨机] 本 ROCm 构建没编 CPU LAPACK ⇒ nn.init.orthogonal_ 的 geqrf 在 CPU 上直接报错。
    做法: 就地搬到 DEVICE 用 torch 自己的 orthogonal_(算法与符号约定原样), 再搬回 CPU。
    副作用是好的: gaussian 改由 CUDA 生成器抽 ⇒ 后面任何可选分支(PL/task_cond)消耗 CPU RNG
    都不再错开正交阵的随机流, A/B 两臂共享参数的初值因此严格一致。"""
    with torch.no_grad():
        td = t.detach().to(DEVICE)
        nn.init.orthogonal_(td, gain)
        t.data.copy_(td.to("cpu"))
    return t

class VisualEmbed(nn.Module):
    """单 patch 16×16像素 → 768 → 256d, 正交+球面 (M1 原样, 逐 patch 调用)"""
    def __init__(self, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(768, out_dim, bias=False)
        _orth_init(self.proj.weight)
        if int(args.pure_learn_ops) >= 2:
            self.radius = nn.Parameter(torch.tensor(1.0))   # 球面半径原本被 normalize 钉成 1.0 ⇒ 可学, 初值1.0
    def forward(self, img):
        v = F.normalize(self.proj(img), dim=-1)
        return v if getattr(self, 'radius', None) is None else self.radius * v

class I2IGenM2(nn.Module):
    """M2 生成头: M1 v4.1 逐patch机制 (batch=b×16) + P7跨patch上下文 + 因式分解位置编码"""
    def __init__(self, embed_dim=256, n_patch=NPATCH, n_in=N_IN, n_bins=args.n_bins):
        super().__init__()
        be = torch.empty(n_bins, embed_dim); _orth_init(be)
        self.register_buffer("bin_embed", be)                      # 冻结视觉词典
        if int(args.pure_learn_ops) >= 2:
            # [PL 二级] 冻结词典是最后也是最大的一个非自学习件: 它同时是输入查表和输出分类器
            # (16 个颜色格被一组固定正交基钉死, CE 的梯度到它就被丢弃)。初值逐位不变。
            self.register_buffer("bin_embed_0", be.clone())
            del self.bin_embed
            self.bin_embed = nn.Parameter(be)
            self.attr_w = nn.Parameter(torch.tensor(1.0))           # patch_tokens = pe + attr_w·attr_proj(·) 的常数 1.0
        self.register_buffer("avg3", self._neighbor_avg(PATCH, 1)) # patch 内 3×3
        self.register_buffer("avg5", self._neighbor_avg(PATCH, 2)) # patch 内 5×5
        self.mix_in = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mix_c3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mix_c5 = nn.Linear(embed_dim, embed_dim, bias=False)
        for m in (self.mix_in, self.mix_c3, self.mix_c5):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
        self.pctx_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # P7 word_out → patch上下文
        nn.init.xavier_uniform_(self.pctx_proj.weight, gain=0.1)
        self.attr_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # E2: per-patch P3Vis → patch_token
        nn.init.xavier_uniform_(self.attr_proj.weight, gain=0.1)
        self.input_proj = nn.Linear(512, embed_dim, bias=False)    # sent_vec + P3Vis
        self.encoder = nn.Sequential(nn.Linear(embed_dim, args.encoder_width), nn.GELU(),
                                     nn.Linear(args.encoder_width, embed_dim), nn.GELU())
        for m in self.encoder:
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)
        self.patch_pos = nn.Parameter(torch.randn(n_patch, embed_dim) * 0.1)
        self.inpatch_pos = nn.Parameter(torch.randn(n_in, embed_dim) * 0.1)
        self.gate = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        self.gate_pair = nn.Linear(embed_dim * 2, embed_dim, bias=False)  # [缺陷修复] gate吃(patch,位内位置)交互
        nn.init.xavier_uniform_(self.gate_pair.weight, gain=0.1)
        if args.spatial_upgrade:
            # [空间升级 S1] 8方向线状移位上下文: 边缘/纹理方向可学习(固定均值只会"糊")
            self.shifts = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            self.register_buffer("shiftM", self._shift_matrix(PATCH, self.shifts))   # [8,768,768]
            self.mix_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.mix_a3 = nn.Linear(embed_dim, embed_dim, bias=False)
            self.mix_a5 = nn.Linear(embed_dim, embed_dim, bias=False)
            self.mix_d = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False)
                                        for _ in range(len(self.shifts))])   # block-diagonal 折叠
            for m in [self.mix_q, self.mix_a3, self.mix_a5] + list(self.mix_d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            # [空间升级 S2] patch 内邻域注意力: 输出位←输入位的 768×768 关系矩阵 = 可检视的"像素间规律"
            self.attn_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.attn_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.attn_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.attn_o = nn.Linear(embed_dim, embed_dim, bias=False)
            for m in (self.attn_q, self.attn_k, self.attn_v, self.attn_o):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
        if args.transform_v2:
            # [转换多样性 v2] 四件套（详见迭代日志）：无缝全图邻域 / 自学习路由 / 多头注意力 / copy-transform 门控
            offs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (2, -2), (-2, 2), (-2, -2)]
            idx = torch.full((len(offs), NPATCH * N_IN), 0, dtype=torch.long)
            msk = torch.zeros(len(offs), NPATCH * N_IN)
            for k, (dy, dx) in enumerate(offs):
                for p in range(NPATCH * N_IN):
                    patch_i, rem = divmod(p, N_IN)
                    y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
                    c = rem % 3
                    py, px = divmod(patch_i, 4)
                    Y, X = py * PATCH + y_in + dy, px * PATCH + x_in + dx
                    if 0 <= Y < IMG and 0 <= X < IMG:      # 跨 patch 的真邻居（无缝）
                        idx[k, p] = ((Y // 16) * 4 + X // 16) * N_IN + (Y % 16) * 48 + (X % 16) * 3 + c
                        msk[k, p] = 1.0
            self.register_buffer("nbr_idx", idx)           # [16,12288] 无缝邻居索引(越界=自身)
            self.register_buffer("nbr_mask", msk)          # [16,12288] 越界权重置零
            self.nbr_lin = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in offs])
            for m in self.nbr_lin: nn.init.xavier_uniform_(m.weight, gain=0.1)
            self.router = nn.Linear(256, len(offs), bias=False)   # 自学习路由: 每位按内容软选 16 支路
            nn.init.xavier_uniform_(self.router.weight, gain=0.1)
            self.router_temp = nn.Parameter(torch.tensor(1.0))    # router 温度 (自项修复配套)
            self.attn_q4 = nn.Linear(256, 256, bias=False)        # 多头 patch 内注意力 4×64
            self.attn_k4 = nn.Linear(256, 256, bias=False)
            self.attn_v4 = nn.Linear(256, 256, bias=False)
            self.attn_o4 = nn.Linear(256, 256, bias=False)
            for m in (self.attn_q4, self.attn_k4, self.attn_v4, self.attn_o4):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            self.tgate = nn.Linear(256, 1, bias=False)            # copy/transform 显式门控
            nn.init.xavier_uniform_(self.tgate.weight, gain=0.1)
            # [自学习·门加宽 2026-10-30] 逐任务门偏置: 让 copy/transform 门按任务各自学基线
            # (去噪该抄、遮补该变换、上色介于两类; 一个全局标量门无法同时表达)
            if args.gate_task:
                self.tgate_task = nn.Embedding(3, 1); nn.init.zeros_(self.tgate_task.weight)
            # 注意力残差的增益(可学): 这一路原本是唯一"满增益1.0无约束"的注入路
            # (实测 E1 attn_out std=0.000 → E2=43.4 → 合成logits std 140~172 → copy门 g 一周期内塌到0)
            self.attn_gain = nn.Parameter(torch.tensor(float(args.v2_attn_gain)))
        if args.s_seamless:
            # [自学习·窄融合 2026-10-29] 只搬 v2 唯一胜过 S 的东西: 跨 patch 无缝邻域(全图 gather),
            # 但混合权重由 router 按内容自学, 注入带可学增益(今晚教训: 无界可学量会被学到 0 ⇒ 绑住)
            _offs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
                     (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (2, -2), (-2, 2), (-2, -2)][:int(args.s_seamless)]
            _idx, _msk = self._nbr_table(_offs)
            self.register_buffer("sn_idx", _idx)
            self.register_buffer("sn_mask", _msk)
            self.sn_lin = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in _offs])
            for m in self.sn_lin: nn.init.xavier_uniform_(m.weight, gain=0.1)
            self.sn_router = nn.Linear(256, len(_offs), bias=False)   # 自学: 每位按内容选哪个方向的全局邻居
            nn.init.zeros_(self.sn_router.weight)                      # 初始均匀, 不预设方向
            self.sn_temp = nn.Parameter(torch.tensor(1.0))
            self.sn_gain = nn.Parameter(torch.tensor(0.1))             # 有界注入(初值 0.1, 可学)
        if args.task_cond:
            # [自学习 T1] 任务/σ/mask 从"设计死的不进网络"变成可学条件: 三者零初始化, 训练起点与关闭时完全一致
            self.task_e = nn.Embedding(3, embed_dim); nn.init.zeros_(self.task_e.weight)
            self.sigma_e = nn.Embedding(4, embed_dim); nn.init.zeros_(self.sigma_e.weight)   # 0=无σ, 1..3=σ0.1/0.2/0.3
            self.mask_h = nn.Linear(3, embed_dim, bias=False); nn.init.zeros_(self.mask_h.weight)
        if args.prompt_cond:
            # [G3 正式入口] 提示词有自己的 embedding, 不再借道 mask_h。零初始化 ⇒ 第0步与关闭时逐位相同。
            # [G5 扩展] 槽位数 = 变形名数(5), 吃 multi-hot ⇒ 组合的嵌入是各元件嵌入之和(组合泛化的结构前提)
            self.prompt_e = nn.Embedding(NT, embed_dim); nn.init.zeros_(self.prompt_e.weight)
            self.prompt_attr = nn.Linear(embed_dim, 256, bias=False); nn.init.zeros_(self.prompt_attr.weight)
            # 两条注入路各自的增益: 值域(0,2)、初值恰为1.0 ⇒ 无界可学量会被学到0(手册§11教训3)
            self.cond_gain = nn.Parameter(torch.zeros(1))
            self.attr_gain = nn.Parameter(torch.zeros(1))
        if args.prompt_cond and args.mult_compose:
            # [G12 补的一格] 档轴上的乘子连乘: W[k,c] 是变形名 k 在通道 c 上的 [from_bin, to_bin] logits 迁移表
            # M_c = Π_k (I + sel_k·W[k,c]) ⇒ 两个名同时请求时冒出 W_a·W_b 交叉项, 这正是"嵌入相加"表达不出来的那一格
            # W 零初始化 ⇒ M=I ⇒ 第0步逐位不变; sel_k=0 的因子恰为 I ⇒ 去噪/遮补/未请求名逐位不变(护栏是结构给的)
            # 三种互斥参数化: 默认=自由 16×16 表可学(G12/D臂) | mult_oracle=理想图硬接且冻结(G13/E臂) |
            #                  mult_mix=只许在候选乘子族里学(G14/F臂)，可学的量降到 240+15 个标量
            if args.mult_mix or args.mult_convex:
                # [G14/G15] 参数化收进乘子族: 可学的只剩"16 元候选混合权重"(G15 再去掉 G14 那道 tanh 门)
                self.mixW = nn.Parameter(torch.zeros(NT, 3, 16))
                if args.mult_mix:
                    self.mixGate = nn.Parameter(torch.zeros(NT, 3))
                self.register_buffer("mix_maps", perm_maps(MIX_GRID, args.n_bins))    # [16,nb,nb] 冻结
                self.register_buffer("mix_bias", (torch.arange(16) == 0).float() * CONVEX_BIAS)
            elif args.mult_oracle:
                # [G13] 把这张表交出去, 不让它学: 填理想置换图后冻结 ⇒ "表达"这一格由构造保证, 剩下的差只可能来自别处
                self.bin_gain = nn.Parameter(oracle_bin_maps(args.n_bins))
                self.bin_gain.requires_grad_(False)
            else:
                self.bin_gain = nn.Parameter(torch.zeros(NT, 3, args.n_bins, args.n_bins))
            if args.int_map and not hasattr(self, "mix_maps"):
                # [R9] 借用 G14/G15 那**同一份**冻结候选族构造(perm_maps(MIX_GRID)), 不新造第二种网格 ⇒ 不会两处漂移
                self.register_buffer("mix_maps", perm_maps(MIX_GRID, args.n_bins))
        if args.refl_branch:
            # [G7 补的一格] 水平反射: 镜像不在引擎原有算子集合里(8 个 ±1 移位与 ±2 邻域都是平移), 用 gather 直接给通路
            self.register_buffer("refl_idx", _refl_index())
            self.refl_lin = nn.Linear(embed_dim, embed_dim, bias=False)
            if args.spat_init > 0: nn.init.orthogonal_(self.refl_lin.weight, gain=args.spat_init)
            else: nn.init.zeros_(self.refl_lin.weight)
            self.refl_gain = nn.Parameter(torch.zeros(1))          # 增益 2·sigmoid ⇒ 初值 1.0; 权重零 ⇒ 第0步逐位不变
            # [G9 补的一格] ①环形平移 gather(与镜像同口径的显式置换); ②**由提示词来选算子**:
            #   G7 的教训是算子一直在、增益却被网络压到 0.52 ⇒ 无条件常开的通路只能整体关小。
            #   perm_sel 把 multi-hot 提示词映射成 2 个算子门 α=(镜像,平移), 零初始化 ⇒ 初值 α=0.5, 而 W 零 ⇒ 第0步仍逐位不变。
            self.register_buffer("roll_idx", _roll_index())
            self.roll_lin = nn.Linear(embed_dim, embed_dim, bias=False)
            if args.spat_init > 0: nn.init.orthogonal_(self.roll_lin.weight, gain=args.spat_init)
            else: nn.init.zeros_(self.roll_lin.weight)
            self.roll_gain = nn.Parameter(torch.zeros(1))
            self.perm_sel = nn.Linear(NT, 2, bias=True); nn.init.zeros_(self.perm_sel.weight); nn.init.zeros_(self.perm_sel.bias)
            self.perm_w = nn.Parameter(torch.zeros(2)); self.perm_b = nn.Parameter(torch.zeros(2))   # [R4] 逐算子独立门(零初值⇒α=0.5, 与旧同初值)
        if args.pure_learn_ops:
            # [清一色自学习 PL] 冠军S路上"非自学习"的三类固定件全部换成可学, 初值精确等于原口径:
            # (1) 冻结的邻域均值/移位矩阵 → 可学核, 用 L1 行归一化绑住
            #     (原矩阵每行非负且和为1, 或全0 ⇒ 归一化后逐位不变, 只差 fp32 舍入 ~1e-7)
            self.avg3_p = nn.Parameter(self.avg3.clone())
            self.avg5_p = nn.Parameter(self.avg5.clone())
            self.register_buffer("avg3_0", self.avg3.clone())
            self.register_buffer("avg5_0", self.avg5.clone())
            if args.spatial_upgrade:
                self.shiftM_p = nn.Parameter(self.shiftM.clone())
                self.register_buffer("shiftM_0", self.shiftM.clone())
            # (2) 14 个常数 1.0 加法 → 逐支路门 gate = 1 + tanh(内容头 + 条件头)
            #     两头均零初始化 ⇒ 第0步 gate 恒等于 1.0; 值域 (0,2) 有界(手册§11教训3: 无界可学量会被学到0)
            self.bgate_q = nn.Linear(embed_dim, 14, bias=True)
            nn.init.zeros_(self.bgate_q.weight); nn.init.zeros_(self.bgate_q.bias)
            self.bgate_c = nn.Linear(embed_dim, 14, bias=False)
            nn.init.zeros_(self.bgate_c.weight)
            # (3) gate_pair 原本只吃位置编码 ⇒ 训练完是固定的逐位置查找表, 看不见图/任务/σ
            self.gp_c = nn.Linear(embed_dim, embed_dim, bias=False)
            nn.init.zeros_(self.gp_c.weight)
        self.h_scale = nn.Parameter(torch.tensor(2.0))
        _orth_init(self.input_proj.weight)
        if int(args.ycbcr_head):
            self.ycb_y = nn.Linear(embed_dim, n_bins, bias=False)
            self.ycb_cb = nn.Linear(embed_dim, n_bins, bias=False)
            self.ycb_cr = nn.Linear(embed_dim, n_bins, bias=False)
            for _m in (self.ycb_y, self.ycb_cb, self.ycb_cr):
                _orth_init(_m.weight)                       # 与 bin_embed 同口径(16×256 正交行)
            self.ycb_gate = nn.Linear(embed_dim, 1, bias=False)   # 不带 bias: 这样"投影后除范数"与"归一化输入"逐位等价
            nn.init.zeros_(self.ycb_gate.weight)                  # 初值 G=1+tanh(0)=1 ⇒ 联合路第0步就全强度在场
            self.register_buffer("ycb_ctr", (torch.arange(n_bins, dtype=torch.float32) + 0.5) / n_bins)
            self._ycb_parts = None                          # 三路色 logits 的暂存, 供 run_batch 算辅助 CE
            self._ycb_gate = None                           # 门读数暂存(只有 head>=1 时才有)
            self._ycb_out = 0.0                             # 最近一次 forward 的越界目标比例(仅日志)
        # [R30] 档内连续残差头: 零初始化 ⇒ 初始连续解码==格中心(逐位); 旗标关时模块不存在 ⇒ 参数列表逐位不变。
        if args.resid_head:
            self.resid_head = nn.Linear(256 + int(args.resid_input), 1, bias=True)
            nn.init.zeros_(self.resid_head.weight)
            nn.init.zeros_(self.resid_head.bias)
        # [R49] 卷积精化解码器: 残差式, 末层零初始化 ⇒ 初始 refined=coarse(恒等闸)。
        if args.refine_cnn:
            self.refine_cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1))
            nn.init.zeros_(self.refine_cnn[-1].weight)
            nn.init.zeros_(self.refine_cnn[-1].bias)
        # [R51] 连续RGB像素头: 接P7联动后的q(256d)→1, sigmoid→[0,1]。在多模态架构内, 非外挂。
        if args.pixel_cont:
            self.cont_head = nn.Linear(256, 1, bias=True)
            nn.init.zeros_(self.cont_head.weight)
            with torch.no_grad():
                self.cont_head.bias.fill_(0.0)   # sigmoid(0)=0.5 中性起点

    # ══════════ [架构·色度因子化输出头] 依据是实测, 不是设想 ══════════
    # 冠军存点(65.20%)上量的: 目标 Cb 的 bin7+bin8 两格吃掉 80.0% 像素(熵 1.93/4.00, 仅 8/16 格 >1%),
    # 而 R 用满 16/16 格(熵 3.34) ⇒ 16 格几乎全花在亮度上, 色度的有效分辨率≈2档;
    # 上色因此平在 47~61 且加监督(color_aux E200 只 +0.26)、跑满 400 轮都不动;
    # n_bins 全局加倍(32)早测出负结果 54.24 —— 与本解释自洽: 要的是把格子重分给色度, 不是到处加格子。
    # 做法: 同一个位内特征 q → Y/Cb/Cr 三个 16 阶头(与 bin_embed 同初始化口径);
    #       期望值经**固定精确**的逆 YCbCr 合成期望 RGB, "到各 RGB bin 中心的负平方距离"作联合 logits,
    #       逐位有界可学门 G=1+tanh(·)(初值 0 ⇒ G=1, 全强度在场) 加到原逐通道 logits 上。
    #       分数口径不变(仍 16 bin RGB argmax) ⇒ 与 65.23 / S1 / S2 直接可比。
    #       越界目标钳到端格 ⇒ 联合路对"少见的浓色"退化成端点, 由越界比例的打印来暴露这件事。
    def _ycb_targets(self, target_bin):
        """目标 RGB bin → (Y,Cb,Cr) 的 bin 目标(逐像素, 形状[...,256]) + 越界比例。纯确定性, 不含任何随机数。
        ★256 不是笔误: Y/Cb/Cr 是**像素**级量(一个像素一个亮度/两个色度), 而 logits/target_bin 是**通道位**级(768)。
          辅助 CE 用 repeat_interleave(3) 把像素目标铺到它自己的 3 个通道位上(见 run_batch), 这里保持像素级最干净。"""
        nb = int(args.n_bins); span = float(args.chroma_span)
        ctr = (target_bin.to(torch.float32) + 0.5) / nb                       # [b,16,768] 格中心
        r = ctr[..., 0::3]; g = ctr[..., 1::3]; b = ctr[..., 2::3]             # 位序 c=idx%3 ⇒ [b,16,256]
        Y = 0.299 * r + 0.587 * g + 0.114 * b
        Cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
        Cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
        y_t = torch.clamp((Y * nb).floor(), 0, nb - 1).long()
        cb_t = torch.clamp(((Cb - (0.5 - span)) / (2 * span) * nb).floor(), 0, nb - 1).long()
        cr_t = torch.clamp(((Cr - (0.5 - span)) / (2 * span) * nb).floor(), 0, nb - 1).long()
        out = ((Cb < 0.5 - span) | (Cb > 0.5 + span) | (Cr < 0.5 - span) | (Cr > 0.5 + span))
        return y_t, cb_t, cr_t, float(out.float().mean())

    def _ycb_sweep(self, target_bin):
        """[量程扫描] 同一批目标在若干 span 下的越界比例, 让 span 的取值是量出来的而不是拍的。"""
        nb = int(args.n_bins)
        ctr = (target_bin.to(torch.float32) + 0.5) / nb
        r = ctr[..., 0::3]; g = ctr[..., 1::3]; b = ctr[..., 2::3]
        Cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
        Cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
        o = []
        for sp in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            f = float(((Cb < 0.5 - sp) | (Cb > 0.5 + sp) | (Cr < 0.5 - sp) | (Cr > 0.5 + sp)).float().mean())
            o.append(f"span{sp:.2f}→越界{f*100:.1f}% 格宽{2*sp/nb:.4f}")
        return " | ".join(o)

    def _ycb_joint(self, q, inv):
        """q [b,16,768,256] → (联合RGB logits[b,16,768,16], 三路色 logits)。
        ★三个色头吃的是 **归一化后的 q**：直接吃原始 q 时实测初值 CE = Y 5.24 / Cb 7.77 / Cr 8.20(合计 21.2)
          而主 CE 只有 1.99 ⇒ 辅助项反过来绑架梯度(比值 10.66)。归一化后初值 CE 应落在 ln(16)=2.77 附近,
          这才是"可学量带球面归一化"这条规矩的作用点(手册§11教训3)。
        ★实现用 `(W·q)·inv` 而不是 `W·(q·inv)`：三个头都没有 bias ⇒ 两者逐位等价,
          但后者要实体化一个与 q 同形的张量(bs16 时实测正是那块 192.0 MiB 的分配)。"""
        nb = int(args.n_bins); span = float(args.chroma_span)
        ly = self.ycb_y(q) * inv; lcb = self.ycb_cb(q) * inv; lcr = self.ycb_cr(q) * inv   # [b,16,768,16]
        ctr = self.ycb_ctr
        Y = F.softmax(ly, dim=-1) @ ctr                                        # ∈[0,1]
        lo = 0.5 - span
        Cb = lo + (F.softmax(lcb, dim=-1) @ ctr) * (2 * span)
        Cr = lo + (F.softmax(lcr, dim=-1) @ ctr) * (2 * span)
        R = Y + 1.402 * (Cr - 0.5)
        G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
        B = Y + 1.772 * (Cb - 0.5)
        _ch = torch.arange(q.shape[2], device=q.device) % 3                    # 每个位在 R/G/B 中的哪一个
        _sel = torch.stack((R, G, B), dim=-1).gather(-1, _ch.view(1, 1, -1, 1).expand_as(Y[..., None]))[..., 0]
        _d = (_sel[..., None] - ctr) * nb                                      # 偏离多少"档"
        return torch.clamp(-0.5 * _d * _d, min=-8.0), (ly, lcb, lcr)

    def _gain(self, p):
        return 2.0 * torch.sigmoid(p)
    def _attr_inject(self, vis_attr, prompt_id):
        """[G3 第二条路] 同一份 prompt_e 进 P3 attr 侧(整图属性256维, input_proj 之前): 零初始化 ⇒ 第0步逐位不变"""
        if not args.prompt_cond or prompt_id is None:
            return vis_attr
        # [R12] pval 可能宽于 NT(尾部带几何参数) ⇒ 进嵌入表前切回前 NT 维, 参数维不许扰动"名"的表示
        return vis_attr + self._gain(self.attr_gain) * self.prompt_attr(prompt_id[:, :NT] @ self.prompt_e.weight)

    def _bin_gain_eff(self):
        """当前参数化下"每(名,通道)的档轴迁移**偏差表**" W[k,c]（_bin_mult_M 吃的是 I+sel·W 里的那个 W）"""
        if args.mult_mix or args.mult_convex:
            lg = MIX_BETA * (self.mixW + self.mix_bias) if args.mult_convex else MIX_BETA * self.mixW
            wgt = torch.softmax(lg, dim=-1)                                       # [NT,3,16] 混合权重
            cand = torch.einsum('kcg,glm->kclm', wgt, self.mix_maps)               # [NT,3,nb,nb] Σ w_g·P(m_g)
            eye = torch.eye(cand.shape[-1], device=cand.device, dtype=cand.dtype)
            if args.mult_convex:
                return cand - eye      # [G15] 无门: 请求该名 ⇒ 因子=cand(凸组合 ⇒ 元素≥0、行和=1)，没请求 ⇒ 因子恰为 I
            return torch.tanh(self.mixGate).view(NT, 3, 1, 1) * (cand - eye)        # [G14] gate=0 ⇒ W=0 ⇒ M 逐位=I
        return self.bin_gain

    def _Pm_exact(self, m):
        """[R33] 精确档轴图: 与 to_bins 同一个量化器 floor(m·ctr·nb), one-hot, 无插值。
        m=1 ⇒ idx=j ⇒ P=I（恒等逐位保持）。形状与 _Pm 输出一致。"""
        nb = args.n_bins
        dev, dt = m.device, m.dtype
        ctr = (torch.arange(nb, device=dev, dtype=dt) + 0.5) / nb
        idx = (m.unsqueeze(-1) * ctr).clamp(0.0, 1.0 - 1e-6)
        idx = (idx * nb).long().clamp(0, nb - 1)
        return F.one_hot(idx, nb).to(dt)

    def _int_map_M(self, pval):
        """[R9] 构造式参数化档轴图。pval[b,k]: 激活维=该行强度 m, 其余=0。
           做法: 由 m 直接算出它在 MIX_GRID 上的**连续格号** gf=-6*log2(m)(网格就是 2^(-g/6), 与跑前自检同源),
           取相邻两格 pm[floor(gf)]/pm[floor(gf)+1] 按小数部分线性插值 ⇒ P(m); W=P(m)-I; 再按名连乘。
           不读 bin_gain/mixW ⇒ 这条规则不进学习, 词表外的 m 由构造覆盖。
           MULT_TABLE[k,c]==1 的通道不动(偏红的 R); 几何名强度=1.0 ⇒ P(1)=I ⇒ 不碰空间算子(与 G13 自检第4条同口径)。"""
        pm = self.mix_maps                                              # [G,nb,nb] 冻结
        G, nb, _ = pm.shape
        dev, dt = pm.device, pm.dtype
        x = pval.to(device=dev, dtype=dt)[:, :NT]                        # [b,NT] ([R12] 尾部几何参数不参与档轴构造)
        on = (x > 0).to(dt)                                             # 名是否被请求
        xv = torch.where(on > 0, x, torch.ones_like(x))                  # 未请求 ⇒ m=1 ⇒ P=I ⇒ 因子恰为 I
        mt = torch.as_tensor(MULT_TABLE, device=dev, dtype=dt)           # [NT,3]
        eff = torch.where(mt.unsqueeze(0) < 1.0, xv.unsqueeze(-1),
                          torch.ones_like(xv).unsqueeze(-1))             # [b,NT,3] 每(名,通道)的有效乘子
        P = self._Pm_exact(eff) if args.int_exact else self._Pm(eff)     # [b,NT,3,nb,nb] [R33]
        eye = torch.eye(nb, device=dev, dtype=dt)
        M = eye.view(1, 1, nb, nb).expand(x.shape[0], 3, nb, nb).clone()
        for k in range(NT):
            M = M @ (eye + (P[:, k] - eye) * on[:, k].view(-1, 1, 1, 1))
        if args.chan_mult and pval.shape[1] >= PV_W:
            # [R16] 逐通道乘子族: M[b,c] 再右乘 P(m_c)。名维全零时上面的连乘恰为 I ⇒ 结果就是纯参数给出的规则。
            cm = pval[:, NT + 11:NT + 14].to(device=dev, dtype=dt)
            act = cm.abs().sum(1) > 0
            if bool(act.any()):
                cm = torch.where(act.view(-1, 1), cm, torch.ones_like(cm))   # 未请求的行给恒等, 免得 0 被 clamp 到网格末端
                _Pcm33 = self._Pm_exact(cm) if args.int_exact else self._Pm(cm)   # [R33]
                M = torch.where(act.view(-1, 1, 1, 1), M @ _Pcm33, M)
        return M

    def _Pm(self, eff):
        """[R9→R16] 乘子 → 档轴置换图 P(m) 的**唯一实现**: 在冻结网格 MIX_GRID(=2^(-g/6)) 上按 log2 线性插值。
           名族(_int_map_M)与逐通道族(chan_mult)都走这里 ⇒ 不会出现"两处各自硬编码再漂移"(G12 那批读数栽过的坑)。
           入 eff 任意形状, 出 eff.shape + (nb,nb)。"""
        pm = self.mix_maps
        G = pm.shape[0]
        gf = (-6.0 * torch.log2(eff.clamp(float(MIX_GRID[-1]), 1.0))).clamp(0.0, G - 1.0)
        gi = gf.long().clamp(0, G - 2)
        t = (gf - gi.to(gf.dtype)).clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        return (1.0 - t) * pm[gi] + t * pm[gi + 1]

    def _bin_mult_M(self, prompt_id):
        """[G12] M[b,c,from,to] = Π_k (I + mh[b,k]·bin_gain[k,c]) —— 变形名在档轴上的连乘算子"""
        if args.int_map:
            return self._int_map_M(prompt_id)                # [R9] 构造式: 由给定强度直接出图, 不读 bin_gain
        W = self._bin_gain_eff()                                 # [NT,3,nb,nb]
        nb = W.shape[-1]
        eye = torch.eye(nb, device=W.device, dtype=W.dtype)
        mh = prompt_id.float().to(W.device).t()                   # [NT,b]
        M = eye.expand(1, W.shape[1], nb, nb).repeat(mh.shape[1], 1, 1, 1)   # [b,3,nb,nb]
        for k in range(NT):
            M = M @ (eye + mh[k].view(-1, 1, 1, 1) * W[k])        # sel_k=0 的因子恰为 I
        return M

    def _roll_tbl_for(self, dY, dX, device):
        """[R12] 按请求的位移**现造** patch-major 环形平移索引表(向量化, 不落 Python 逐位循环)。
           口径与 _roll_index(dY,dX) 逐字相同: out[Y,X] = in[(Y-dY)%IMG, (X-dX)%IMG], 跑前阳性对照逐位核。
           同一个 (dY,dX) 只造一次, 之后走缓存(考试 119 行也只会用到几种位移)。"""
        key = (int(dY), int(dX), str(device))
        # [R12 坑] 缓存键必须带 device: 跑前阳性对照会在 CPU 上造同一张表核对 numpy, 只按 (dY,dX) 缓存
        #   会把 CPU 张量还给 CUDA 路径 ⇒ index_select 直接 RuntimeError(第一次跑就栽在这里)
        _c = getattr(self, "_roll_cache", None)
        if _c is None: _c = self._roll_cache = {}
        if key in _c: return _c[key]
        q = torch.arange(NPATCH * N_IN, device=device)
        patch_i, rem = q // N_IN, q % N_IN
        y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
        c = rem % 3
        py, px = patch_i // 4, patch_i % 4
        Y = (py * PATCH + y_in - key[0]) % IMG
        X = (px * PATCH + x_in - key[1]) % IMG
        idx = ((Y // 16) * 4 + X // 16) * N_IN + (Y % 16) * 48 + (X % 16) * 3 + c
        # [R19] 表缓存加 LRU 上限: R14/R16 的参数是连续的(任意仿射系数 / 逐通道乘子),
        #   无界 dict 会随步数无限增长(每张 12288×8B≈96KB)。表由参数确定性构造 ⇒
        #   淘汰后重算逐位相同, 只是多一次 arange 级构造, 不影响任何判决。
        if len(_c) >= TABLE_CACHE_CAP:
            _c.pop(next(iter(_c)))
        _c[key] = idx
        return idx

    def _chan_tbl_for(self, perm, device):
        """[R15] 按通道置换 π **现造** patch-major 位置轴 gather 表: 位置 p 的通道 c=p%3, 它要取的源位是同一像素的 π(c) 通道
           ⇒ idx[p] = p - c + π(c)。像素内 3 个通道位在 patch-major 里是连续的(…+(X%16)*3+c), 所以这就是一个纯位置置换。"""
        key = (tuple(int(v) for v in perm), str(device))
        _c = getattr(self, "_cp_cache", None)
        if _c is None: _c = self._cp_cache = {}
        if key in _c: return _c[key]
        q = torch.arange(NPATCH * N_IN, device=device)
        ch = q % 3
        pi = torch.tensor([int(perm[0]), int(perm[1]), int(perm[2])], device=device, dtype=q.dtype)
        idx = q - ch + pi[ch]
        # [R19] 表缓存加 LRU 上限: R14/R16 的参数是连续的(任意仿射系数 / 逐通道乘子),
        #   无界 dict 会随步数无限增长(每张 12288×8B≈96KB)。表由参数确定性构造 ⇒
        #   淘汰后重算逐位相同, 只是多一次 arange 级构造, 不影响任何判决。
        if len(_c) >= TABLE_CACHE_CAP:
            _c.pop(next(iter(_c)))
        _c[key] = idx
        return idx

    def _apply_chan_map(self, logits, prompt_id):
        """[R15] 构造式通道置换: 作用在输出的位置轴上, 与 R14 的仿射族同一套机制。
           顺序: 先通道置换、再空间算子(_apply_spat_map 在本函数之后调用) ⇒ out = spatial(chanperm(pred))。
           没给参数 ⇒ 逐位原样返回。注意它**不看名维**: 这一格要验的就是"规则只靠参数能不能成立"。"""
        if not (args.prompt_cond and args.chan_perm) or prompt_id is None or prompt_id.shape[1] < PV_W:
            return logits
        pv = prompt_id[:, NT + 8:NT + 11].to(logits.device)
        act = pv.abs().sum(1) > 0
        if not bool(act.any()):
            return logits
        b, P_, W_, NB_ = logits.shape
        out = logits.reshape(b, P_ * W_, NB_)
        perms = [tuple(int(round(float(pv[i, j]) * 2)) for j in range(3)) for i in range(b)]
        new = out
        for pm in sorted({perms[i] for i in range(b) if bool(act[i])}):
            if sorted(pm) != [0, 1, 2]:
                w(f"[响亮警告] 通道置换参数 {(pm)} 不是 0/1/2 的一个排列 ⇒ 该请求无意义, 已跳过(不是静默当成恒等)")
                continue
            rows = torch.tensor([i for i in range(b) if bool(act[i]) and perms[i] == pm],
                                device=out.device, dtype=torch.long)
            if rows.numel() == 0: continue
            new = new.clone() if new is out else new
            new[rows] = out[rows].index_select(1, self._chan_tbl_for(pm, out.device))
        return new.reshape(b, P_, W_, NB_)

    def _affine_tbl_for(self, af, device):
        """[R14] 按仿射系数**现造** patch-major gather 表: idx[p] = 输出位 p 应取的源位下标。
           输出口径与 _roll_index/_refl_index 逐字相同(patch-major: ((Y//16)*4+X//16)*N_IN+(Y%16)*48+(X%16)*3+c),
           源坐标 Ys=round(a·Y+b·X+e)%IMG, Xs=round(c·Y+d·X+f)%IMG。
           ★单位约定(跑前自检就是被这个坑拦下的): af 六个数**一律像素单位**, numpy 的 apply_affine 同口径;
             ÷IMG 只存在于指示向量的线格式里, 解码在 _apply_spat_map 做。两处各用一种单位就会差出 0.85 这种大数。"""
        key = (tuple(round(float(v), 6) for v in af), str(device))
        _c = getattr(self, "_aff_cache", None)
        if _c is None: _c = self._aff_cache = {}
        if key in _c: return _c[key]
        a, b, c, d, e, f = (float(v) for v in af)
        q = torch.arange(NPATCH * N_IN, device=device)
        patch_i, rem = q // N_IN, q % N_IN
        y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
        ch = rem % 3
        py, px = patch_i // 4, patch_i % 4
        Y = (py * PATCH + y_in).to(torch.float32)
        X = (px * PATCH + x_in).to(torch.float32)
        Ys = torch.round(a * Y + b * X + e).to(torch.long) % IMG
        Xs = torch.round(c * Y + d * X + f).to(torch.long) % IMG
        idx = ((Ys // 16) * 4 + Xs // 16) * N_IN + (Ys % 16) * 48 + (Xs % 16) * 3 + ch
        # [R19] 表缓存加 LRU 上限: R14/R16 的参数是连续的(任意仿射系数 / 逐通道乘子),
        #   无界 dict 会随步数无限增长(每张 12288×8B≈96KB)。表由参数确定性构造 ⇒
        #   淘汰后重算逐位相同, 只是多一次 arange 级构造, 不影响任何判决。
        if len(_c) >= TABLE_CACHE_CAP:
            _c.pop(next(iter(_c)))
        _c[key] = idx
        return idx

    def _apply_spat_map(self, logits, prompt_id):
        """[R11] 构造式空间算子: 把请求的镜像/平移**直接作用在输出的位置轴**上。
           logits [b,NPATCH,N_IN,nb]; 位置轴展平后与 refl_idx/roll_idx 同口径(patch-major, G7/G9 已逐位核过)。
           顺序与 apply_combo 一致: 先镜像后平移 ⇒ 组合请求下 out = roll(flip(pred))。
           没请求空间名 ⇒ 逐位原样返回(第0步与关闭时相同的口径)。"""
        if not (args.prompt_cond and args.spat_map) or prompt_id is None:
            return logits
        mh = prompt_id[:, 3:5].to(logits.device)                     # (翻转, 平移) 指示位
        # [R35] 纯参数平移: 名位没置但 pval 带了位移参数也要触发(与 chan/仿射族的参数触发同口径)。
        #   修的是假接线: 组合行名维全零时位移参数被静默跳过, 目标移了预测没移(R34 探针三角揪出)。
        _shp = prompt_id[:, NT:NT + 2].to(logits.device) if (args.roll_param and prompt_id.shape[1] > NT) else None
        _sh_act = (_shp.abs().sum(1) > 0) if _shp is not None else None
        # [R14] 仿射参数(若指示向量够宽): 有参数的行由仿射表接管, 旧的两张表不再作用于它们
        _af = prompt_id[:, NT + 2:NT + 8].to(logits.device) if prompt_id.shape[1] >= PV_W else None
        if _af is not None and args.affine_map:
            _use_af = _af.abs().sum(1) > 0                            # [b] 该行是否请求了仿射
        else:
            _af = None; _use_af = None
        if not bool((mh > 0).any()) and (_use_af is None or not bool(_use_af.any())) \
                and (_sh_act is None or not bool(_sh_act.any())):
            return logits
        b, P_, W_, NB_ = logits.shape
        out = logits.reshape(b, P_ * W_, NB_)
        if _use_af is not None and bool(_use_af.any()):
            mh = mh * (~_use_af).to(mh.dtype).view(b, 1)              # 仿射行退出旧的镜像/平移路, 免得被作用两次
        if bool((mh[:, 0] > 0).any()):                                  # 镜像: 无参数, 固定那张表
            _g = out.index_select(1, self.refl_idx)
            out = torch.where((mh[:, 0] > 0).view(b, 1, 1), _g, out)
        if bool((mh[:, 1] > 0).any()) or (_sh_act is not None and bool(_sh_act.any())):   # 平移: [R12] 位移量按行现造表 [R35] 参数也触发
            _sh = _shp
            if _sh is None:
                _dYdX = [(8, 8)] * b
            else:
                _dYdX = [(int(round(float(_sh[i, 0]) * IMG)), int(round(float(_sh[i, 1]) * IMG))) for i in range(b)]
                # 约定: 两维都 0 = 没给参数 ⇒ 回落到 (8,8)(与 R11 逐位相同), 免得"没写参数"被当成"位移0"
                _dYdX = [(8, 8) if v == (0, 0) else v for v in _dYdX]
            _new = out
            for _s in sorted(set(_dYdX)):
                _rows = torch.tensor([i for i, v in enumerate(_dYdX) if v == _s and (
                                         float(mh[i, 1]) > 0 or (_sh_act is not None and bool(_sh_act[i])))],
                                     device=out.device, dtype=torch.long)
                if _rows.numel() == 0: continue
                _t = self.roll_idx if _s == (8, 8) else self._roll_tbl_for(_s[0], _s[1], out.device)
                _new = _new.clone() if _new is out else _new
                _new[_rows] = out[_rows].index_select(1, _t)
            out = _new
        if _use_af is not None and bool(_use_af.any()):
            # [R14] 逐"不同参数组"取表现造 ⇒ 一个批次里可以混着竖直翻转/旋转90°/任意仿射, 各自按自己的系数走
            def _dec14(row):
                v = [round(float(x), 6) for x in row]
                # 线格式里 e/f 是 ÷IMG 存的 ⇒ 这里乘回像素单位(与 _affine_tbl_for/apply_affine 同口径)
                return (v[0], v[1], v[2], v[3], round(v[4] * IMG, 4), round(v[5] * IMG, 4))
            _afs = [_dec14(_af[i]) for i in range(b)]
            _new = out
            for _a in sorted({_afs[i] for i in range(b) if bool(_use_af[i])}):
                _rows = torch.tensor([i for i in range(b) if bool(_use_af[i]) and _afs[i] == _a],
                                     device=out.device, dtype=torch.long)
                if _rows.numel() == 0: continue
                _t = self._affine_tbl_for(_a, out.device)
                _new = _new.clone() if _new is out else _new
                _new[_rows] = out[_rows].index_select(1, _t)
            out = _new
        return out.reshape(b, P_, W_, NB_)

    def apply_rules(self, logits, prompt_id):
        """[R18] 规则层唯一入口：档轴乘子 → 通道置换 → 空间算子，顺序固定。

        R17 教训：三个构造式算子原先各挂一条前向，导致 R9/R11/R12/R14/R15/R16
        在 --transform_v2 下全部静默失效。此后任何产出 logits 的前向都必须调用本方法
        （模块启动时的 [R18 接线自检] 会用 AST 静态核对，漏接当场 exit 3）。
        未请求任何规则时三个算子都是逐位恒等 ⇒ 本方法对 logits 逐位无影响。
        """
        logits = self._apply_bin_mult(logits, prompt_id)
        logits = self._apply_chan_map(logits, prompt_id)
        logits = self._apply_spat_map(logits, prompt_id)
        # [R30] 残差当 K=1 伪 logits 过**同一套**位置轴算子(不含档轴乘子) ⇒ 与 logits 同表 gather, 零漂移。
        _r30 = getattr(self, '_resid_raw', None)
        if _r30 is not None:
            self._resid_raw = None
            _r4 = _r30.unsqueeze(-1)
            _r4 = self._apply_chan_map(_r4, prompt_id)
            _r4 = self._apply_spat_map(_r4, prompt_id)
            self._resid_parts = _r4.squeeze(-1)
        return logits

    def _apply_bin_mult(self, logits, prompt_id):
        """把 M 作用到 logits 的档轴: out[...,m] = Σ_l logits[...,l]·M[c(from pos),l,m]"""
        if not (args.prompt_cond and args.mult_compose) or prompt_id is None:
            return logits
        M = self._bin_mult_M(prompt_id)                            # [b,3,nb,nb]
        ch = torch.arange(logits.shape[2], device=M.device) % 3
        return torch.einsum('bpwl,bwlm->bpwm', logits, M[:, ch])   # [b,16,768,16]
    def _cond_vec(self, task_id, sigma_val, mask_feat, prompt_id=None):
        c = None
        if args.task_cond:
            if task_id is not None:
                c = self.task_e(task_id)
            if sigma_val is not None:
                sb = torch.clamp((sigma_val * 10.0).round().long(), 0, 3)
                e = self.sigma_e(sb)
                c = e if c is None else c + e
            if mask_feat is not None:
                e = self.mask_h(mask_feat)
                c = e if c is None else c + e
        if args.prompt_cond and prompt_id is not None:
            # [G3] 提示词是条件向量的正式成员, 与 mask 摘要互不重叠
            # [G5] multi-hot [b,5] × 嵌入表 = 各变形名嵌入之和 ⇒ 未见组合落在已学元件的线性组合上
            e = prompt_id[:, :NT] @ self.prompt_e.weight                 # [R12] 切回前 NT 维(尾部是几何参数)
            c = e if c is None else c + e
        return c
    @staticmethod
    def _rownorm(M):
        """L1 行归一化: 每行权重总量钉在 1(允许带符号)。原 box/shift 矩阵行和已为1或全0 ⇒ 初值逐位不变"""
        return M / M.abs().sum(-1, keepdim=True).clamp(min=1e-6)
    @staticmethod
    def _nbr_table(offs):
        """全图邻居索引/掩码: 跨 patch 的真邻居(越界处 index 指向自身但 mask=0)"""
        idx = torch.full((len(offs), NPATCH * N_IN), 0, dtype=torch.long)
        msk = torch.zeros(len(offs), NPATCH * N_IN)
        for k, (dy, dx) in enumerate(offs):
            for p in range(NPATCH * N_IN):
                patch_i, rem = divmod(p, N_IN)
                y_in, x_in = rem // (PATCH * 3), (rem % (PATCH * 3)) // 3
                c = rem % 3
                py, px = divmod(patch_i, 4)
                Y, X = py * PATCH + y_in + dy, px * PATCH + x_in + dx
                if 0 <= Y < IMG and 0 <= X < IMG:
                    idx[k, p] = ((Y // 16) * 4 + X // 16) * N_IN + (Y % 16) * 48 + (X % 16) * 3 + c
                    msk[k, p] = 1.0
        return idx, msk
    @staticmethod
    def _neighbor_avg(img, r):
        n = img * img * 3
        M = torch.zeros(n, n)
        for y in range(img):
            for x in range(img):
                for y2 in range(max(0, y - r), min(img, y + r + 1)):
                    for x2 in range(max(0, x - r), min(img, x + r + 1)):
                        for c in range(3):
                            M[(y * img + x) * 3 + c, (y2 * img + x2) * 3 + c] = 1.0
        return M / M.sum(dim=1, keepdim=True)
    @staticmethod
    def _shift_matrix(img, shifts):
        """[K,768,768] 移位矩阵: S_k[p,q]=1 当 q 是 p 沿方向 k 的同通道邻居(越界补零)"""
        n = img * img * 3
        Ms = torch.zeros(len(shifts), n, n)
        for si, (dy, dx) in enumerate(shifts):
            for y in range(img):
                for x in range(img):
                    y2, x2 = y + dy, x + dx
                    if 0 <= y2 < img and 0 <= x2 < img:
                        for c in range(3):
                            Ms[si, (y * img + x) * 3 + c, (y2 * img + x2) * 3 + c] = 1.0
        return Ms

    def forward(self, deg_bins, sent_vec, vis_attr, patch_ctx, task_id=None, sigma_val=None, mask_feat=None, prompt_id=None):
        if args.transform_v2:
            return self._forward_v2(deg_bins, sent_vec, vis_attr, patch_ctx, deg_bins.shape[0],
                                    task_id, sigma_val, mask_feat, prompt_id)
        """deg_bins [b,16,768] | sent_vec [b,256] P7句向量 | vis_attr [b,256] | patch_ctx [b,16,256] P7 word_out"""
        b = deg_bins.shape[0]
        q_in = self.bin_embed[deg_bins.reshape(b * NPATCH, N_IN)]           # [b*16,768,256]
        c = self._cond_vec(task_id, sigma_val, mask_feat, prompt_id) \
            if (args.task_cond or args.prompt_cond) else None   # [PL] 提前算: 门和身份都要吃
        vis_attr = self._attr_inject(vis_attr, prompt_id)       # [G3 第二条路] 提示词进 P3 attr 侧
        if args.pure_learn_ops:
            a3, a5 = self._rownorm(self.avg3_p), self._rownorm(self.avg5_p)
            _gr = self.bgate_q(q_in)                                        # [b*16,768,14]
            if c is not None:
                _gr = _gr + self.bgate_c(c).repeat_interleave(NPATCH, 0).unsqueeze(1)
            G = 1.0 + float(args.gate_span) * torch.tanh(_gr)               # 初值恒 1.0(与 span 无关), 值域 (1-span, 1+span)
            g = [G[..., i:i + 1] for i in range(14)]
        else:
            a3, a5 = self.avg3, self.avg5
        ctx3 = a3 @ q_in
        ctx5 = a5 @ q_in
        if args.spatial_upgrade:
            sM = self._rownorm(self.shiftM_p) if args.pure_learn_ops else self.shiftM
            _gw = (lambda i, t: g[i] * t) if args.pure_learn_ops else (lambda i, t: t)
            # S1: 逐支路 Linear 相加 (= concat+Linear 的 block-diagonal 折叠, 免 8GB cat 峰值)
            feat = _gw(0, self.mix_q(q_in)) + _gw(1, self.mix_a3(ctx3)) + _gw(2, self.mix_a5(ctx5))
            for k in range(len(self.shifts)):
                feat = feat + _gw(3 + k, self.mix_d[k](sM[k] @ q_in))
            if args.s_seamless:
                # 自学无缝邻域: 全图 gather(跨 patch 接缝可见) × router 自学方向权重 × 有界增益
                qg = q_in.reshape(b, NPATCH * N_IN, 256)
                w = torch.softmax(self.sn_router(qg) / self.sn_temp.clamp(min=0.1), dim=-1)   # [b,12288,K]
                seam = None
                for k in range(len(self.sn_lin)):
                    gk = qg.index_select(1, self.sn_idx[k]) * self.sn_mask[k].view(1, -1, 1)
                    term = w[..., k:k + 1] * self.sn_lin[k](gk)
                    seam = term if seam is None else seam + term
                feat = feat + self.sn_gain * seam.reshape(b * NPATCH, N_IN, 256)
                if getattr(self, 'seam_probe', False):
                    with torch.no_grad():
                        ent = -(w * (w + 1e-9).log()).sum(-1)                # [b,12288] 方向选择熵(上限 lnK)
                        self._seam_stats = dict(gain=float(self.sn_gain), ent=float(ent.mean()),
                                                ent_max=float(np.log(len(self.sn_lin))),
                                                wmax=float(w.max(-1).values.mean()),
                                                seam=float(seam.std()),
                                                w=[round(float(x), 3) for x in w.mean((0, 1))])
            # S2: 分块邻域注意力 (查询位 128 一块, 免 768×768 全量分数矩阵)
            k_proj = self.attn_k(q_in + self.inpatch_pos.unsqueeze(0))      # 输入内容+位置 作 K
            v_proj = self.attn_v(q_in)
            outs = []
            for i in range(0, N_IN, 128):
                sl = slice(i, min(N_IN, i + 128))
                qk = self.attn_q(feat[:, sl, :]) @ k_proj.transpose(-1, -2) / (256 ** 0.5)
                outs.append(torch.softmax(qk, dim=-1) @ v_proj)             # [n,ch,768]→[n,ch,256]
            feat = feat + _gw(11, self.attn_o(torch.cat(outs, dim=1)))      # 像素转换: 从输入位聚合
        else:
            feat = self.mix_in(q_in) + self.mix_c3(ctx3) + self.mix_c5(ctx5)  # [b*16,768,256]
        feat = feat.reshape(b, NPATCH, N_IN, 256)
        if args.refl_branch:
            # [G7] 反射项不吃 14 门(与 s_seamless 同口径: 新增通路自带增益, 不动旧门的语义)
            # [G9] ①再加一条环形平移 gather(与镜像同一口径的显式置换); ②两条各乘一个**由提示词决定**的门 α:
            #   G7 实测常开通路被网络整体压到增益 0.521 ⇒ 不区分"要不要这个算子"的通路只能被关掉。零初始化 W ⇒ 第0步仍逐位不变。
            _qg = q_in.reshape(b, NPATCH * N_IN, 256)
            if prompt_id is not None and args.comp_gate:
                _mh = prompt_id.float()[:, 3:5]                                                    # (flip, roll) 指示位
                _al = torch.sigmoid(self.perm_w.unsqueeze(0) * _mh + self.perm_b.unsqueeze(0))     # [R4] 逐算子独立门: pair=两门同开, 不相消
            else:
                _al = torch.sigmoid(self.perm_sel(prompt_id.float()[:, :NT])) if prompt_id is not None \
                    else torch.full((b, 2), 0.5, device=q_in.device, dtype=q_in.dtype)          # [b,2]=(镜像,平移)门
            _rf = self.refl_lin(_qg.index_select(1, self.refl_idx)).reshape(b, NPATCH, N_IN, 256)
            _rl = self.roll_lin(_qg.index_select(1, self.roll_idx)).reshape(b, NPATCH, N_IN, 256)
            feat = feat + (self._gain(self.refl_gain) * _al[:, 0].view(b, 1, 1, 1) * _rf
                           + self._gain(self.roll_gain) * _al[:, 1].view(b, 1, 1, 1) * _rl)
        G4 = G.reshape(b, NPATCH, N_IN, 14) if args.pure_learn_ops else None
        _pctx = self.pctx_proj(patch_ctx).unsqueeze(2)                      # [b,16,1,256] 广播到 patch 内 768 位
        feat = feat + (G4[..., 12:13] * _pctx if args.pure_learn_ops else _pctx)
        _hin_raw = self.encoder(self.input_proj(torch.cat([sent_vec, vis_attr], dim=-1)))
        if c is not None:
            # [G3 修量级] 旧写法在归一化**之前** h_in+c: 实测 ‖h_in‖≈2.79e5 而 ‖c‖≈3.5
            #   ⇒ 条件在单位身份上只占 1.3e-5, 被残差流幅度淹没(与《说明》:995 同病)。
            #   先把身份归到单位球面再按有界增益加条件 ⇒ 注入占比 = 增益×‖c‖, 不再恒为 0.000。
            h_in = F.normalize(_hin_raw, dim=-1) \
                + (self._gain(self.cond_gain) if hasattr(self, "cond_gain") else 1.0) * c
        else:
            h_in = _hin_raw
        h = self.h_scale * F.normalize(h_in, dim=-1)                        # [b,256] 球面
        if args.gate_pair:
            # [缺陷修复] 门控吃 (patch, 位内位置) 拼接交互 → 恢复 (图像×绝对位置) 完整记忆通路。
            gp_raw = self.gate_pair(torch.cat(
                [self.patch_pos.unsqueeze(1).expand(-1, N_IN, -1),
                 self.inpatch_pos.unsqueeze(0).expand(NPATCH, -1, -1)], dim=-1))
            if args.pure_learn_ops and c is not None:
                # gp 原本"输入无关"(只吃位置) ⇒ 训练完是固定的逐位置查找表, 看不见图/任务/σ; PL 让它也吃条件
                gp_raw = gp_raw.unsqueeze(0) + self.gp_c(c).view(b, 1, 1, 256)
                hp_in = h.view(b, 1, 1, 256) * torch.tanh(torch.clamp(gp_raw, -2.0, 2.0))
            else:
                gp = torch.tanh(torch.clamp(gp_raw, -2.0, 2.0))
                hp_in = h.unsqueeze(1).unsqueeze(2) * gp.unsqueeze(0)       # [b,16,768,256]
            if args.pure_learn_ops: hp_in = G4[..., 13:14] * hp_in
            q = feat + hp_in + self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0)
        else:
            hp_in = h.unsqueeze(1) * torch.tanh(torch.clamp(
                self.gate(self.inpatch_pos), -2.0, 2.0)).unsqueeze(0)       # [b,768,256] 旧版: 丢patch交互
            q = (feat + hp_in.unsqueeze(1)
                    + self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0))
        if args.pure_learn_ops and getattr(self, 'pl_probe', False):
            # [PL探针] 三件必须同时看的量: 门学到多少(偏离1.0) / 固定核学到多少(偏离初值) / 条件占身份多大比(‖c‖÷‖h_in‖)
            # 第三项补的是 T1 探针的缺口: 只打三张表各自范数, 分不清"学到"和"两个大向量反向抵消"
            with torch.no_grad():
                self._pl_stats = dict(
                    g_mean=[round(float(G[..., i].mean()), 4) for i in range(14)],
                    g_dev=float((G - 1.0).abs().mean()), g_min=float(G.min()), g_max=float(G.max()),
                    d_a3=float((a3 - self.avg3_0).norm() / self.avg3_0.norm().clamp(min=1e-9)),
                    d_a5=float((a5 - self.avg5_0).norm() / self.avg5_0.norm().clamp(min=1e-9)),
                    d_sM=float((sM - self.shiftM_0).norm() / self.shiftM_0.norm().clamp(min=1e-9)),
                    h_scale=float(self.h_scale),
                    c_norm=float(c.norm(dim=-1).mean()) if c is not None else 0.0,
                    hin_norm=float(h_in.norm(dim=-1).mean()),
                    hin_raw_norm=float(_hin_raw.norm(dim=-1).mean()),
                    cond_gain=float(self._gain(self.cond_gain)) if hasattr(self, "cond_gain") else 0.0,
                    attr_gain=float(self._gain(self.attr_gain)) if hasattr(self, "attr_gain") else 0.0,
                    prompt_c=float(self.prompt_e.weight.norm()) if args.prompt_cond else 0.0)
                _pt = {}
                if task_id is not None:                     # 按任务读门: 总分掩盖任务反转, 只有拆开才知道门在替谁做主
                    _tr = task_id.repeat_interleave(NPATCH)
                    for t in range(3):
                        _m = _tr == t
                        if bool(_m.any()):
                            _pt[t] = [round(float(G[_m, ..., i].mean()), 4) for i in range(14)]
                self._pl_stats["per_task"] = _pt
                if int(args.pure_learn_ops) >= 2:
                    # 词典可学的代价是它可能塌了(两个颜色格学成同一个) ⇒ 用"离初值正交基多远"盯住
                    _g = F.normalize(self.bin_embed, dim=-1) @ F.normalize(self.bin_embed_0, dim=-1).t()
                    self._pl_stats["d_be"] = float((self.bin_embed - self.bin_embed_0).norm()
                                                   / self.bin_embed_0.norm().clamp(min=1e-9))
                    self._pl_stats["be_dev"] = float((_g - torch.eye(_g.shape[0], device=_g.device)).abs().max())
                    self._pl_stats["attr_w"] = float(self.attr_w)
        if args.resid_head:
            _q31 = q
            if args.resid_input:
                _q31 = torch.cat([q, self._resid_src], dim=-1)        # [R31] +连续输入值
            self._resid_raw = self.resid_head(_q31).squeeze(-1)       # [R30] [b,16,768]
        if args.pixel_cont:
            # [R52] 头只重建内容: 几何用构造gather搬q特征, 颜色用构造乘子。不让头学搬移。
            _b52 = q.shape[0]
            if prompt_id is not None:
                _idx52 = self._geo_index(prompt_id)
                _m52 = self._color_mult(prompt_id)
            else:
                _idx52 = torch.arange(NPATCH * N_IN, device=q.device).unsqueeze(0).expand(_b52, -1)
                _m52 = torch.ones(_b52, 3, device=q.device)
            _qg52 = q.reshape(_b52, -1, 256).gather(1, _idx52.unsqueeze(-1).expand(-1, -1, 256)).reshape(_b52, NPATCH, N_IN, 256)
            _c52 = torch.sigmoid(self.cont_head(_qg52)).reshape(_b52, -1)
            _ci52 = _pm_to_img(_c52) * _m52[:, None, None, :]
            self._refined = _img_to_pm(_ci52.clamp(0.0, 1.0))   # [b,12288] 构造规则后的连续RGB
        logits = torch.einsum('bpwd,kd->bpwk', q, self.bin_embed)             # [b,16,768,16]
        if args.comp_probe:
            self._pre_logits = logits                                               # [R34]
        logits = self.apply_rules(logits, prompt_id)                            # [R18] 规则层唯一入口(顺序不变)
        if int(args.ycbcr_head):
            # 归一化按 1/‖q‖ 乘在**投影结果**上(标量场 [b,16,768,1]), 不实体化归一化的 q:
            # 后者在 bs16 下正好是那块 192.0 MiB 的分配, 而三个头与门都是无 bias 线性 ⇒ 逐位等价。
            _inv = q.pow(2).sum(-1, keepdim=True).clamp(min=1e-12).rsqrt()     # [b,16,768,1]
            _jl, self._ycb_parts = self._ycb_joint(q, _inv)
            # 门也必须按归一化输入算: 原始 q 幅度极大(实测身份范数量级 1e7~1e10),
            # 直接进 tanh 会恒贴 ±1 ⇒ G 恒等于 0 或 2, 机制一开始就被冻在极端(§8.3-d "门会贴饱和"的真正成因)
            _g = 1.0 + torch.tanh(self.ycb_gate(q) * _inv)                     # [b,16,768,1] 初值恒 1.0
            if int(args.ycbcr_head) >= 2:
                # head=2 = **真的把输出字母换掉**: 完全丢开逐通道 base 分类器, 只留联合路。
                # head=1 只是把联合项加在旧 logits 上 ⇒ argmax 仍然由逐 RGB bin 分类器决定,
                # 12 轮实测两点上色同向偏负(−0.89/−0.37)与"字母表没真换"自洽:  reachable 集合没变。
                logits = _g * _jl
            else:
                logits = logits + _g * _jl
            with torch.no_grad():   # 门读数: 不打印它就无法区分"这条路没用"与"网络把它关了"
                _gm = float(_g.mean()); _gs = float(_g.std())
                _near = float(((_g < 0.05) | (_g > 1.95)).float().mean())
                self._ycb_gate = dict(g_mean=_gm, g_sd=_gs, g_min=float(_g.min()), g_max=float(_g.max()),
                                      sat=_near, mode=int(args.ycbcr_head),
                                      jl_absmax=float(_jl.abs().max()))
        return logits

    def _forward_v2(self, deg_bins, sent_vec, vis_attr, patch_ctx, b, task_id=None, sigma_val=None, mask_feat=None, prompt_id=None):
        """[转换多样性 v2] 无缝全图邻域 gather + 自学习路由 + 多头 patch 内注意力 + copy/transform 显式门控"""
        q_flat = self.bin_embed[deg_bins.reshape(b, NPATCH * N_IN)]         # [b,12288,256] 输入查表(全图)
        # 1) 无缝邻居分支 × 自学习路由: 每位按输入内容在 16 个空间支路间软选择(跨 patch 真邻居, 无缝)
        router = torch.softmax(self.router(q_flat) / self.router_temp.clamp(min=0.1), dim=-1)  # [b,12288,16]
        acc = torch.zeros_like(q_flat)
        for k in range(self.nbr_idx.shape[0]):
            g = q_flat.index_select(1, self.nbr_idx[k]) * self.nbr_mask[k].unsqueeze(0).unsqueeze(-1)
            acc = acc + router[..., k:k + 1] * self.nbr_lin[k](g)
        acc = acc + q_flat   # [修复] (0,0) 自项: 邻居支路之外保留当前像素内容, copy 门才学得出来
        q4 = acc.reshape(b, NPATCH, N_IN, 256)
        # 2) 多头 patch 内注意力(4×64): 位→位的细粒度关系, 每头可学不同规律(颜色延续/边缘/纹理周期)
        Qp, Kp, Vp = self.attn_q4(q4), self.attn_k4(q4), self.attn_v4(q4)
        if args.v2_sdpa:
            # [R24/P1] SDPA 路: 头布局=末维连续 64 块(与旧路切片同序), scale 默认 1/sqrt(64)=旧式 /64**0.5。
            #   不物化 [b,16,768,768] ⇒ bs8 全量口径的 OOM 根因消除; 数值差仅 fp32 结合序(等价闸把门)。
            def _h24(x):
                return x.reshape(b * NPATCH, N_IN, 4, 64).transpose(1, 2)
            _o24 = F.scaled_dot_product_attention(_h24(Qp), _h24(Kp), _h24(Vp))
            attn_out = self.attn_o4(_o24.transpose(1, 2).reshape(b, NPATCH, N_IN, 256))
        else:
            outs = []
            for hi in range(4):
                sl = slice(hi * 64, (hi + 1) * 64)
                s = Qp[..., sl] @ Kp[..., sl].transpose(-1, -2) / (64 ** 0.5)   # [b,16,768,768]
                outs.append(torch.softmax(s, dim=-1) @ Vp[..., sl])
            attn_out = self.attn_o4(torch.cat(outs, dim=-1))
        q4 = q4 + self.attn_gain * attn_out        # 可学增益(初值 --v2_attn_gain, 默认1.0=原口径)
        # 3) 全局身份 + 逐位门控(同 v1)
        vis_attr = self._attr_inject(vis_attr, prompt_id)     # [G3 第二条路] 与 S 路口径一致
        _hin_raw = self.encoder(self.input_proj(torch.cat([sent_vec, vis_attr], dim=-1)))
        h_in = _hin_raw
        if args.task_cond or args.prompt_cond:
            c = self._cond_vec(task_id, sigma_val, mask_feat, prompt_id)
            if c is not None:
                h_in = F.normalize(_hin_raw, dim=-1) \
                    + (self._gain(self.cond_gain) if hasattr(self, "cond_gain") else 1.0) * c
        h = self.h_scale * F.normalize(h_in, dim=-1)
        gp = torch.tanh(torch.clamp(self.gate_pair(torch.cat(
            [self.patch_pos.unsqueeze(1).expand(-1, N_IN, -1),
             self.inpatch_pos.unsqueeze(0).expand(NPATCH, -1, -1)], dim=-1)), -2.0, 2.0))
        hp = h.unsqueeze(1).unsqueeze(2) * gp.unsqueeze(0)
        q4 = q4 + hp \
                + self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0)
        synth = torch.einsum('bpwd,kd->bpwk', q4, self.bin_embed)           # 合成路 logits
        # 4) copy/transform 显式门控: 每个像素位自己决定"抄输入"还是"用合成"
        pos_full = (self.patch_pos.unsqueeze(1) + self.inpatch_pos.unsqueeze(0)).reshape(1, NPATCH * N_IN, 256)
        copy_l = ((q_flat + pos_full) @ self.bin_embed.t()).reshape(b, NPATCH, N_IN, 16)
        _gt = self.tgate(q4)
        if getattr(self, 'tgate_task', None) is not None and task_id is not None:
            _gt = _gt + self.tgate_task(task_id.to(q4.device)).view(b, 1, 1, 1)   # 逐任务偏置(初始 0)
        g = torch.sigmoid(_gt).reshape(b, NPATCH, N_IN, 1)
        out = g * synth + (1.0 - g) * copy_l
        if args.resid_head:
            _q31 = q4
            if args.resid_input:
                _q31 = torch.cat([q4, self._resid_src], dim=-1)       # [R31] +连续输入值
            self._resid_raw = self.resid_head(_q31).squeeze(-1)       # [R30]
        if getattr(self, 'v2_probe', False):        # 分支级探针: 只在 --v2_probe_every 命中的那一步算
            with torch.no_grad():
                r_ent = -(router * (router + 1e-9).log()).sum(-1)                  # [b,12288] 路由熵(上限ln16=2.77)
                self._v2_stats = dict(
                    r_ent=float(r_ent.mean()), r_max=float(router.max(-1).values.mean()),
                    br=[round(float(x), 3) for x in router.mean((0, 1))],
                    g=float(g.mean()), g_sd=float(g.std()),
                    s_sd=float(synth.std()), c_sd=float(copy_l.std()),
                    q_sd=float(q_flat.std()), acc_sd=float(acc.std()),
                    attn_sd=float(attn_out.std()), hp_sd=float(hp.std()),
                    gain=float(self.attn_gain),
                    used=int(torch.unique(out.argmax(-1)).numel()))
        # [R17 接线补齐] 三个构造式规则算子原先只挂在 S 路(forward:1478-1480) ⇒ 开 --transform_v2 1 时
        #   R9/R11/R12/R14/R15/R16 六个修复**全部静默失效**(在场但不生效, 与代码里 _ycb_warned 那句"v2 路径未接"同类坑)。
        #   v2 的 out 与 S 路 logits 同形 [b,NPATCH,N_IN,nb] ⇒ 同序接上即可(未请求时三个都是逐位恒等, 不影响 v2 原口径)。
        if args.comp_probe:
            self._pre_logits = out                                                  # [R34]
        out = self.apply_rules(out, prompt_id)                           # [R18] 规则层唯一入口(顺序不变)
        return out

# [R18 接线自检] R17 的教训: 三个构造式算子原先只挂在 S 路, v2 路静默失效, 靠人审计才发现。
#   此处用 AST 静态核对所有产出 logits 的前向都必须经过规则层唯一入口 apply_rules；
#   以后新增第三条前向若漏接, 启动时当场 exit 3, 不再等到跑完一轮看分数才发现。
import ast as _ast
import inspect as _inspect
_cls_node = _ast.parse(_inspect.getsource(I2IGenM2)).body[0]
_fwd_nodes = [_f for _f in _cls_node.body
              if isinstance(_f, _ast.FunctionDef)
              and (_f.name == "forward" or _f.name.startswith("_forward_"))]
_missing_wiring = [_f.name for _f in _fwd_nodes
                   if not any(isinstance(_n, _ast.Call)
                              and isinstance(_n.func, _ast.Attribute)
                              and _n.func.attr == "apply_rules"
                              for _stmt in _f.body for _n in _ast.walk(_stmt))]
if _missing_wiring:
    w("[R18 接线自检] FAIL 以下前向未接规则层: %s ⇒ 停在这里, 不开训" % _missing_wiring)
    sys.exit(3)
w("[R18 接线自检] OK %d 条前向均经 apply_rules: %s" % (
    len(_fwd_nodes), [_f.name for _f in _fwd_nodes]))

visual_embed = VisualEmbed().to(DEVICE)
gen = I2IGenM2().to(DEVICE)
p7 = P7WordRouter2048(word_dim=256, inner_dim=256, heads=16, head_dim=16, max_len=128, num_groups=4).to(DEVICE)
n_params = sum(p.numel() for m in [visual_embed, gen, p7] for p in m.parameters())
w(f"[模型] 可训参数: {n_params:,} + 冻结词典{args.n_bins}×256 | P7 跨patch路由已接入 | 全FP32")
# [R23] 跑前配置契约闸: 每个开着的算子旗标, 它依赖的运行时结构必须已在场。
#   §25 那类「旗标开了但依赖没注册」的崩溃从训练中途/自检深处提前到这里, 响亮 exit 2。
#   清单只列有注册依赖的算子; chan_perm/affine_map 的表是按需构造, 无结构依赖。
_contract_need = []
if args.int_map or args.chan_mult:
    _contract_need.append(("int_map/chan_mult", "mix_maps"))
if args.spat_map or args.roll_param:
    _contract_need.append(("spat_map/roll_param", "refl_idx"))
_contract_miss = [(flag, attr) for flag, attr in _contract_need if not hasattr(gen, attr)]
if _contract_miss:
    w("[R23 配置契约] FAIL 缺依赖: %s ⇒ 停在这里, 不开训(检查开关自动补链)" % _contract_miss)
    sys.exit(2)
if _contract_need:
    w("[R23 配置契约] OK %s" % [f"{f}→{a}" for f, a in _contract_need])
# [R24] SDPA 等价闸: 同一随机 q4, 旧显式矩阵路 vs SDPA 路, 差只许 fp32 结合序量级。
#   不过 ⇒ exit 3 不开训(等价性是先决条件, 不是跑完再看)。
if args.transform_v2 and args.v2_sdpa:
    with torch.no_grad():
        _g24 = torch.Generator(device=DEVICE).manual_seed(24)
        _q24 = torch.randn(2, NPATCH, N_IN, 256, generator=_g24, device=DEVICE)
        _Q24, _K24, _V24 = gen.attn_q4(_q24), gen.attn_k4(_q24), gen.attn_v4(_q24)
        _o24 = []
        for _hi in range(4):
            _sl = slice(_hi * 64, (_hi + 1) * 64)
            _s24 = _Q24[..., _sl] @ _K24[..., _sl].transpose(-1, -2) / (64 ** 0.5)
            _o24.append(torch.softmax(_s24, dim=-1) @ _V24[..., _sl])
        _old24 = gen.attn_o4(torch.cat(_o24, dim=-1))
        def _h24g(x):
            return x.reshape(2 * NPATCH, N_IN, 4, 64).transpose(1, 2)
        _new24 = F.scaled_dot_product_attention(_h24g(_Q24), _h24g(_K24), _h24g(_V24))
        _new24 = gen.attn_o4(_new24.transpose(1, 2).reshape(2, NPATCH, N_IN, 256))
        _d24 = float((_old24 - _new24).abs().max())
    w(f"[R24 等价闸] 旧路 vs SDPA max_abs_diff={_d24:.3e} (判据≤1e-4)")
    if _d24 > 1e-4:
        w("[R24 等价闸] FAIL ⇒ 停在这里, 不开训"); sys.exit(3)
# [R25] 跨族组合跑前闸: 退化(单族==现成函数逐位) + 恒等乘子阳性对照 +
#   纯置换组合行 one-hot→apply_rules→argmax 与 numpy 目标逐位(构造性核对, 不等考试打分)。
if args.exam_unseen_combo:
    with torch.no_grad():
        _r25 = np.random.default_rng(25)
        _img25 = _r25.random((IMG, IMG, 3)).astype(np.float32)
        _cm25 = (1.00, 0.85, 0.70); _pm25 = (2, 1, 0)
        _af25 = (0.0, 1.0, -1.0, 0.0, 0.0, float(IMG - 1)); _sh25 = (12, 12)
        _d25a = float(np.abs(apply_cross_combo(_img25, None, None, _cm25, None, None, None) - apply_chan_mult(_img25, _cm25)).max())
        _d25b = float(np.abs(apply_cross_combo(_img25, None, None, None, _pm25, None, None) - apply_chan_perm(_img25, _pm25)).max())
        _d25c = float(np.abs(apply_cross_combo(_img25, None, None, None, None, None, _sh25) - apply_combo_m(_img25, mhot([4]), None, _sh25)).max())
        _d25d = float(np.abs(apply_cross_combo(_img25, None, None, (1.0, 1.0, 1.0), _pm25, _af25, None) - apply_cross_combo(_img25, None, None, None, _pm25, _af25, None)).max())
        w(f"[R25 组合闸] 退化 cm/pm/sh = {_d25a:.3e}/{_d25b:.3e}/{_d25c:.3e} | 恒等乘子 {_d25d:.3e} (判据全 0)")
        if max(_d25a, _d25b, _d25c, _d25d) > 0.0:
            w("[R25 组合闸] FAIL ⇒ 停在这里, 不开训"); sys.exit(3)
        _tgt25 = apply_cross_combo(_img25, None, None, None, _pm25, _af25, None)
        _bin25 = to_bins(to_patch_major(_img25).reshape(-1))
        _lg25 = F.one_hot(torch.as_tensor(_bin25, device=DEVICE), _NB).float().reshape(1, NPATCH, N_IN, _NB)
        _pv25 = torch.zeros(1, PV_W, device=DEVICE)
        _pv25[0, NT + 8:NT + 11] = torch.tensor(_pm25, device=DEVICE, dtype=torch.float32) / 2.0
        _pv25[0, NT + 2:NT + 6] = torch.tensor(_af25[:4], device=DEVICE, dtype=torch.float32)
        _pv25[0, NT + 6] = _af25[4] / IMG; _pv25[0, NT + 7] = _af25[5] / IMG
        _pred25 = gen.apply_rules(_lg25, _pv25).argmax(-1).reshape(-1)
        _tb25 = torch.as_tensor(to_bins(to_patch_major(_tgt25).reshape(-1)), device=DEVICE)
        _n25 = int((_pred25 != _tb25).sum())
        w(f"[R25 构造闸] 纯置换组合 模型路 vs numpy 失配格数={_n25} (判据=0)")
        if _n25 != 0:
            w("[R25 构造闸] FAIL ⇒ 停在这里, 不开训"); sys.exit(3)
# [R30] 残差头跑前闸: 零初始化恒零 + 位置编码残差场与 logits 同表 gather 的构造性核对。
if args.resid_head:
    with torch.no_grad():
        _g30 = torch.Generator(device=DEVICE).manual_seed(30)
        _q30 = torch.randn(2, NPATCH, N_IN, 256 + int(args.resid_input), generator=_g30, device=DEVICE)
        _z30 = float(gen.resid_head(_q30).abs().max())
        w(f"[R30 初始闸] 零初始化残差 max={_z30:.3e} (判据=0)")
        if _z30 != 0.0:
            w("[R30 初始闸] FAIL ⇒ 停在这里, 不开训"); sys.exit(3)
        _idx30 = torch.arange(NPATCH * N_IN, device=DEVICE, dtype=torch.float32)
        gen._resid_raw = _idx30.view(1, NPATCH, N_IN).expand(2, -1, -1).contiguous()
        _lg30 = F.one_hot((_idx30.long() % _NB), _NB).float().view(1, NPATCH, N_IN, _NB).expand(2, -1, -1, -1).contiguous()
        _pv30 = torch.zeros(2, PV_W, device=DEVICE)
        _pv30[:, 4] = 1.0
        _pv30[:, NT] = 12.0 / IMG; _pv30[:, NT + 1] = 12.0 / IMG
        _pv30[:, NT + 8:NT + 11] = torch.tensor([2.0, 1.0, 0.0], device=DEVICE) / 2.0
        _o30 = gen.apply_rules(_lg30, _pv30)
        _rp30 = gen._resid_parts
        _u30 = torch.unique(_rp30[0])
        _perm_ok = (_u30.numel() == NPATCH * N_IN) and bool((_u30.sort().values == _idx30).all())
        _pat_ok = bool(((_rp30.reshape(2, -1).long() % _NB) == _o30.argmax(-1).reshape(2, -1)).all())
        w(f"[R30 对齐闸] 置换完整={_perm_ok} 源位模式一致={_pat_ok} (判据均 True)")
        if not (_perm_ok and _pat_ok):
            w("[R30 对齐闸] FAIL ⇒ 停在这里, 不开训"); sys.exit(3)
# [R33] 精确构造闸: M 必须与 numpy 理想 one-hot 图逐位相同(同量化器), 空请求必须恒等。
if args.int_exact:
    with torch.no_grad():
        _ctr33 = (np.arange(_NB, dtype=np.float32) + 0.5) / _NB
        def _ideal33(mv):
            _ix = np.clip((np.asarray(mv, dtype=np.float32) * _ctr33 * _NB).astype(np.int64), 0, _NB - 1)
            return torch.as_tensor(np.eye(_NB, dtype=np.float32)[_ix], device=DEVICE)
        _d33 = 0.0
        for _m33 in (0.40, 0.55, 0.70, 0.85, 0.5):
            _pv33 = torch.zeros(1, PV_W, device=DEVICE)
            _pv33[0, 0] = _m33                                   # 偏红名, 取 G 通道(R 通道乘子=1 不动)
            _M33 = gen._int_map_M(_pv33)[0, 1]   # M=[b,3,nb,nb], 取 G 通道(偏红的 R 通道乘子=1 不动)
            _d33 = max(_d33, float((_M33 - _ideal33(_m33)).abs().max()))
        _pv33c = torch.zeros(1, PV_W, device=DEVICE)
        _cm33 = (0.70, 0.85, 1.00)
        _pv33c[0, NT + 11:NT + 14] = torch.tensor(_cm33, device=DEVICE)
        _Mc33 = gen._int_map_M(_pv33c)[0]
        _d33c = max(float((_Mc33[c] - _ideal33(_cm33[c])).abs().max()) for c in range(3))
        _d33e = float((gen._int_map_M(torch.zeros(1, PV_W, device=DEVICE))[0]
                       - torch.eye(_NB, device=DEVICE).unsqueeze(0).expand(3, -1, -1)).abs().max())
        w(f"[R33 精确闸] 名通道 max差={_d33:.3e} | 逐通道乘子 max差={_d33c:.3e} | 空请求={_d33e:.3e} (判据全 0)")
        if max(_d33, _d33c, _d33e) > 0.0:
            w("[R33 精确闸] FAIL ⇒ 停在这里, 不开训"); sys.exit(3)
# [R35] 参数触发构造闸: 名维全零 + 位移参数 + 乘子参数 ⇒ 模型路必须逐位等于机械定义
#   roll(to_bins(m·ctr[to_bins(x)]))（接线正确性）; 与真目标 to_bins(m·roll(x)) 的差是量化不交换, 只报不判。
if args.spat_map and args.roll_param and args.chan_mult and args.int_exact:
    with torch.no_grad():
        _r35 = np.random.default_rng(35)
        _img35 = _r35.random((IMG, IMG, 3)).astype(np.float32)
        _cm35 = np.asarray((1.00, 0.85, 0.70), dtype=np.float32)
        _ctr35 = (np.arange(_NB, dtype=np.float32) + 0.5) / _NB
        _imgb35 = to_bins(_img35.reshape(-1))
        _ch35 = np.arange(IMG * IMG * 3) % 3
        _mech35 = to_bins(_cm35[_ch35] * _ctr35[_imgb35]).reshape(IMG, IMG, 3)
        _mech35 = np.roll(_mech35, (12, 12), axis=(0, 1))
        _lg35 = F.one_hot(torch.as_tensor(to_bins(to_patch_major(_img35).reshape(-1)), device=DEVICE), _NB) \
            .float().reshape(1, NPATCH, N_IN, _NB)
        _pv35 = torch.zeros(1, PV_W, device=DEVICE)
        _pv35[0, NT] = 12.0 / IMG; _pv35[0, NT + 1] = 12.0 / IMG
        _pv35[0, NT + 11:NT + 14] = torch.tensor(_cm35, device=DEVICE)
        _o35 = gen.apply_rules(_lg35, _pv35).argmax(-1).reshape(-1)
        _tb35 = torch.as_tensor(to_patch_major(_mech35).reshape(-1), device=DEVICE)   # _mech35 已是 bin 索引, 不再二次量化
        _n35 = int((_o35 != _tb35).sum())
        _tgt35 = np.clip(np.roll(_img35 * _cm35, (12, 12), axis=(0, 1)), 0, 1)
        _tt35 = torch.as_tensor(to_bins(to_patch_major(_tgt35).reshape(-1)), device=DEVICE)
        _gap35 = int((_o35 != _tt35).sum())
        w(f"[R35 参数触发闸] 纯参数暖调×位移12 vs 机械定义 失配格={_n35} (判据=0) | vs 真目标量化不交换差={_gap35}格(只报不判)")
        if _n35 != 0:
            w("[R35 参数触发闸] FAIL ⇒ 停在这里, 不开训"); sys.exit(3)
if args.mult_compose:
    # [G12 阳性对照 A·跑前写死] 档轴置换这一族**能不能表达**上色目标: 把输入档 j 送到 to_bins(m·ctr_j) 的 16 档映射。
    #   它不看图内容、只看输入档 ⇒ 这个一致率就是本机制的天花板; 同时报"两步连乘"(网络真正拿到的那条)。
    #   两支都要报, 不挑高的说; 天花板明显低于 100% 时, 组合侧涨不上去不能只怪学习。
    _NB = args.n_bins
    _ctr_b = (np.arange(_NB, dtype=np.float32) + 0.5) / _NB
    _binmap = lambda mvec: np.stack([to_bins(mvec[c] * _ctr_b).astype(np.int64) for c in range(3)])
    _chk = {}
    for _ids in ([0], [1], [2], [0, 2], [0, 1]):
        _net = np.ones(3, dtype=np.float32)
        for _k in _ids: _net *= MULT_TABLE[_k]
        _P = _binmap(_net)                                                # 直接图(一张矩阵吃净乘子)
        _hits = _hits_ch = _tot = 0
        for _s in sources:
            _x = _s["rgb"]
            _tb = to_bins(to_patch_major(apply_combo(_x, mhot(_ids))).reshape(-1))
            _ib = to_bins(to_patch_major(_x).reshape(-1))
            _c = np.arange(len(_ib)) % 3
            _hits += int((_P[_c, _ib] == _tb).sum())
            _j = _ib.copy()
            for _k in _ids:                                               # 两步连乘: 每步后再量化一次
                _j = _binmap(MULT_TABLE[_k])[_c, _j]
            _hits_ch += int((_j == _tb).sum())
            _tot += len(_ib)
        _chk[combo_label(mhot(_ids))] = (_hits / _tot, _hits_ch / _tot)
    w("[G12档轴可表达自检] " + f"{len(sources)} 图源 × {NPATCH * N_IN} 位 | "
      + " | ".join(f"{_lb} 直接图{100 * _a:.2f}%/连乘{100 * _b:.2f}%" for _lb, (_a, _b) in _chk.items()))

    if not (args.mult_oracle or args.mult_mix or args.mult_convex or args.int_map):
        # [G12 阳性对照 B·仪器闸] 三条必须在开训前逐位成立, 违一条就停:
        #   (1) W=0 时任意 multi-hot 的 M 恰为单位阵 ⇒ 声明的"第0步逐位不变"是真的, 不是推理出来的
        #   (2) 单位阵过档轴 einsum 后 logits 逐位不变 ⇒ 接线本身不改数值
        #   (3) 把 W 换成确定性非零值, 两个名的 M 与 (I+W_a+W_b) 之差恰等于交叉项 W_a·W_b ⇒ 这条路真的是"连乘",
        #       这正是 G11 实测缺的那一格(嵌入相加只能加不能乘)。改完必须逐位还原成 0, 否则起点被污染。
        with torch.no_grad():
            _I = torch.eye(_NB, device=DEVICE)
            _mh = torch.zeros(4, NT, device=DEVICE)
            for _i, _ids in enumerate(([0], [3], [0, 3], list(range(NT)))):
                for _k in _ids: _mh[_i, _k] = 1.0
            _M0 = gen._bin_mult_M(_mh)
            _d1 = float((_M0 - _I).abs().max())
            _L = (torch.arange(2 * 3 * 3 * _NB, device=DEVICE, dtype=torch.float32) % 9).view(2, 3, 3, _NB)
            _L2 = torch.einsum('bpwl,bwlm->bpwm', _L, _M0[:2][:, torch.arange(3, device=DEVICE) % 3])
            _d2 = float((_L2 - _L).abs().max())
            _bg0 = gen.bin_gain.detach().clone()
            gen.bin_gain.data.copy_((torch.arange(NT * 3 * _NB * _NB, device=DEVICE, dtype=torch.float32)
                                     % 17 / 400.0 - 0.02).view(NT, 3, _NB, _NB))
            _one = lambda ids: gen._bin_mult_M(torch.tensor(mhot(ids), dtype=torch.float32, device=DEVICE).view(1, -1))[:, 0]
            _Ma, _Mb, _Mab = _one([0]), _one([2]), _one([0, 2])
            _cross = float((_Mab - (_I + (_Ma - _I) + (_Mb - _I))).abs().max())
            _cw = float(((_Ma - _I) @ (_Mb - _I)).abs().max())
            gen.bin_gain.data.copy_(_bg0)
            _d4 = float(gen.bin_gain.abs().max())
            w(f"[G12乘子自检] W=0 时 max‖M−I‖={_d1:.6f} | 档轴einsum恒等 maxΔ={_d2:.6f} | "
              f"连乘−相加 max‖差‖={_cross:.6f}(交叉项‖W_a·W_b‖max={_cw:.6f}) | 还原后‖W‖max={_d4:.6f} "
              f"(判据: 前三项 0/0/>0 且还原=0.000000)")
            if _d1 != 0.0 or _d2 != 0.0 or _cross <= 0.0 or _d4 != 0.0:
                w("[自检不符] 乘子连乘这条路没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)
    elif args.int_map:
        # [R9 阳性对照·跑前写死] 构造式档轴图必须由构造保证五件事, 违一条当场停(仪器闸, 不是判负闸):
        #   (1) 网格口径与 gf 公式同源: -6*log2(MIX_GRID) 逐位等于 arange(16)
        #   (2) 没请求任何名 / 请求了但强度=1.0 ⇒ M 逐位=I
        #   (3) 几何名(翻转/平移, 强度恒1.0) ⇒ M 逐位=I(不碰空间算子)
        #   (4) m 恰在格点上 ⇒ M 逐行 one-hot, 且被乘通道那张图逐位等于 mix_maps[g](没被乘的通道=I)
        #   (5) 词表外 m ⇒ 报插值代价: 期望档位移与 to_bins(m*ctr) 偏离>1档的格数(网格分辨率的固有价, 只报不判)
        with torch.no_grad():
            _I9 = torch.eye(_NB, device=DEVICE)
            _mg = torch.as_tensor(MIX_GRID, device=DEVICE, dtype=torch.float32)
            _d1 = float(((-6.0 * torch.log2(_mg)) - torch.arange(16, device=DEVICE, dtype=torch.float32)).abs().max())

            def _pv9(pairs):
                v = torch.zeros(1, NT, device=DEVICE)
                for _k9, _m9 in pairs: v[0, _k9] = _m9
                return gen._int_map_M(v)[0]                            # [3,nb,nb]

            _d2 = float((gen._int_map_M(torch.zeros(2, NT, device=DEVICE)) - _I9).abs().max())
            _d2b = float((_pv9([(0, 1.0), (2, 1.0)]) - _I9).abs().max())
            _d3 = max(float((_pv9([(3, 1.0)]) - _I9).abs().max()), float((_pv9([(4, 1.0)]) - _I9).abs().max()))
            _g6 = int(round(float(-6.0 * np.log2(0.5))))                  # 0.5 恰是格点 g=6
            _M6 = _pv9([(0, 0.5)])
            _hot6 = int(((_M6.abs().sum(-1) - 1.0).abs() > 1e-6).sum())
            _d4 = max(float((_M6[1] - gen.mix_maps[_g6]).abs().max()),      # 偏红的 G/B 通道应为 P(0.5)
                      float((_M6[2] - gen.mix_maps[_g6]).abs().max()),
                      float((_M6[0] - _I9).abs().max()))                    # R 通道应为 I
            _ctr9 = (torch.arange(_NB, device=DEVICE, dtype=torch.float32) + 0.5) / _NB
            _bad9 = 0; _mx9 = 0.0
            for _m9 in (0.40, 0.55, 0.70, 0.85):
                _Mm = _pv9([(2, _m9)])                                    # 变暗: 三通道同乘
                _exp9 = (_Mm * torch.arange(_NB, device=DEVICE, dtype=torch.float32)).sum(-1)
                _tg9 = torch.clamp((_m9 * _ctr9 * _NB).floor(), 0, _NB - 1)
                _dv9 = (_exp9[0] - _tg9).abs()
                _bad9 += int((_dv9 > 1.0).sum()); _mx9 = max(_mx9, float(_dv9.max()))
            # 判据用**明写容差**而不是 ==0: MIX_GRID 是 float32 存的 2^(-g/6), log2 往返不可能逐位回到整数
            #   (第一次跑就是被这个坑拦下的: 实际差 5e-7, 却被 %.6f 打成 0.000000 ⇒ 读数看着全过、闸却 exit 3)。
            #   1e-4 个格号 = 网格步长的万分之一, 远小于任何一档的意义; 打印一律科学计数, 不让 1e-9 冒充 0。
            _T9 = 1e-4
            w(f"[R9构造自检] 网格口径 maxΔ={_d1:.3e} | 空请求/强度1 max‖M−I‖={_d2:.3e}/{_d2b:.3e} | 几何两名 max‖M−I‖={_d3:.3e} "
              f"| m=0.5 落格点: one-hot不合格行={_hot6} 与 mix_maps[6] 最大差={_d4:.3e} "
              f"| 插值代价(m=0.40/0.55/0.70/0.85): 偏离>1档格数={_bad9}/64 最大偏离={_mx9:.2f}档 "
              f"(判据: 前四项均 ≤{_T9:g} 且 one-hot不合格行=0; 插值代价只报不判, 它是网格分辨率的固有价)")
            if max(_d1, _d2, _d2b, _d3, _d4) > _T9 or _hot6:
                w(f"[自检不符] 构造式档轴图没按声明接线(超容差项见上, 容差{_T9:g}) ⇒ 停在这里, 不开训"); sys.exit(3)
    elif args.mult_oracle:
        # [G13 阳性对照·跑前写死] oracle 这条路必须由构造保证五件事, 违一条当场停(仪器闸, 不是判负闸):
        #   (1) 表内容与 oracle_bin_maps 逐位相同  (2) requires_grad=False 且 grad 恒 None ⇒ 它真的不进学习
        #   (3) 每个单元名的 M 是**逐行 one-hot** 的置换阵, 且行映射恰等于 to_bins(m_kc·ctr_j)
        #   (4) 翻转/平移两名(无颜色乘子)的 M 恰为单位阵 ⇒ 硬接不碰空间算子
        #   (5) 7 个组合的 M 仍是置换阵, 且它等于"各名图连乘"——同时报它与"直接图"在 16 档上差多少(量化连乘的固有代价)
        with torch.no_grad():
            _tgt = oracle_bin_maps(_NB).to(gen.bin_gain.device)
            _d1 = float((gen.bin_gain - _tgt).abs().max())
            _rg = bool(gen.bin_gain.requires_grad)
            _I = torch.eye(_NB, device=DEVICE)
            _one = lambda ids: gen._bin_mult_M(torch.tensor(mhot(ids), dtype=torch.float32, device=DEVICE).view(1, -1))
            _bad_map = _bad_hot = 0
            for k in range(NT):
                M = _one([k])[0]                                          # [3,NB,NB]
                _bad_hot += int(((M.abs().sum(-1) - 1.0).abs() > 1e-6).sum())
                _map = M.argmax(-1)                                       # [3,NB]
                _bad_map += int((_map != torch.from_numpy(_binmap(MULT_TABLE[k])).to(M.device)).sum())
            _id_spatial = max(float((_one([3])[0] - _I).abs().max()), float((_one([4])[0] - _I).abs().max()))
            _bad_cmb = _ch_vs_dir = 0
            for _ids in EXAM_COMBOS:
                M = _one(list(_ids))[0]; _map = M.argmax(-1)
                _bad_cmb += int(((M.abs().sum(-1) - 1.0).abs() > 1e-6).sum())
                _net = np.ones(3, dtype=np.float32)
                for _k in _ids: _net *= MULT_TABLE[_k]
                _ch_vs_dir += int((_map != torch.from_numpy(_binmap(_net)).to(M.device)).sum())
            w(f"[G13 oracle自检] 表逐位等于理想图 maxΔ={_d1:.6f} | requires_grad={_rg} | 单元名 one-hot 不合格行={_bad_hot} "
              f"映射不符格={_bad_map}/240 | 空间两名 max‖M−I‖={_id_spatial:.6f} | 组合 one-hot 不合格行={_bad_cmb} | "
              f"连乘图≠直接图的档数={_ch_vs_dir}/336 (判据: 前三项 0/False/0 且空间两名=0.000000)")
            if _d1 != 0.0 or _rg or _bad_hot or _bad_map or _id_spatial != 0.0 or _bad_cmb:
                w("[自检不符] oracle 档轴图没按声明接上 ⇒ 停在这里, 不开训"); sys.exit(3)
    elif args.mult_convex:
        # [G15 阳性对照·跑前写死] 真凸混合要满足五件事, 违一条当场停(仪器闸, 不是判负闸):
        #   (1) 起点是**软恒等**: max‖M−I‖ ≤ 5e-3 —— 这是买"凸性"付的价(G14 那个逐位 0 只能靠允许负权重骗出来)
        #   (2) 概率迁移由构造保证: 混合权重 ≥0 且行和=1；4 种 multi-hot(含"5 名全请求")的 M 不允许出现负元素、行和必须=1
        #   (3) 16 张候选图逐行 one-hot、行和=1、P(m_0)=I(网格构造不许漂移)
        #   (4) 族里含得住目标: 0.55 与 0.7 到最近网格的相对误差 ≤10%
        #   (5) 可学标量 == 240(门已经拿掉) 且**初值就有** ‖∂/∂a‖>0 —— 没有门 ⇒ 活性只能直接量 a，
        #       泛函用 G14 学到的那条教训: 确定性逐位点乘，不能用 abs().sum()(对称抵消会把活性伪装成 0)
        with torch.no_grad():
            _I = torch.eye(_NB, device=DEVICE)
            _mh = torch.zeros(4, NT, device=DEVICE)
            for _i, _ids in enumerate(([0], [3], [0, 3], list(range(NT)))):
                for _k in _ids: _mh[_i, _k] = 1.0
            _d1 = float((gen._bin_mult_M(_mh) - _I).abs().max())
            _d1a = float((gen._bin_mult_M(_mh[:1]) - _I).abs().max())     # 只请求 1 个名(考试里上色格的常态是 1~2 个)
            _d1b = float((gen._bin_mult_M(_mh[3:]) - _I).abs().max())     # 5 名全请求(最坏情况，b 就是为这条定的)
            _Mall = gen._bin_mult_M(torch.ones(1, NT, device=DEVICE))       # 5 名全请求 = 连乘后的净算子
            _neg = int((_Mall < -1e-9).sum())
            _rowsum = float((_Mall.sum(-1) - 1.0).abs().max())
            _wgt = torch.softmax(MIX_BETA * (gen.mixW + gen.mix_bias), -1)
            _wneg = int((_wgt < 0).sum())
            _wsum2 = float((_wgt.sum(-1) - 1.0).abs().max())
            _P = gen.mix_maps
            _hot = int(((_P.abs().sum(-1) - 1.0).abs() > 1e-6).sum())
            _idg0 = float((_P[0] - _I).abs().max())
            _rel = [float(np.min(np.abs(MIX_GRID - t)) / t) for t in (0.55, 0.7)]
            _near = [int(np.argmin(np.abs(MIX_GRID - t))) for t in (0.55, 0.7)]
            _npar = int(gen.mixW.numel())
            _w0 = float(_wgt[..., 0].mean())
        _probe = (torch.arange(_NB * _NB, device=DEVICE, dtype=torch.float32) % 13).view(_NB, _NB)
        _gA = torch.autograd.grad((gen._bin_gain_eff() * _probe).sum(), gen.mixW)[0]
        w(f"[G15 convex自检] 起点 max‖M−I‖={_d1:.6f}(判据 ≤0.005；单名 {_d1a:.6f}/5名全请求 {_d1b:.6f}) 初值 w_0={_w0:.6f} | 全请求 M 负元素={_neg} "
          f"行和不符max={_rowsum:.6f} | 权重 负数={_wneg} 行和偏差={_wsum2:.8f} | 候选 one-hot 不合格行={_hot}/256 "
          f"P(m_0)=I maxΔ={_idg0:.6f} | 0.55/0.7 最近网格 g={_near} 相对误差={_rel[0]*100:.2f}%/{_rel[1]*100:.2f}% | "
          f"可学标量={_npar}(判据 240) | 初值梯度活性 ‖∂/∂a‖={float(_gA.norm()):.4f}(判据 >0)")
        if (_d1 > 5e-3 or _neg or _rowsum > 1e-5 or _wneg or _wsum2 > 1e-6 or _hot or _idg0 != 0.0
                or max(_rel) > 0.1 or _npar != 240 or float(_gA.norm()) <= 0.0):
            w("[自检不符] 真凸混合没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)
        w("  [G15 口径提示] 凸臂起点不是逐位 I 而是软恒等(b=1.25 ⇒ w_0=0.999319，单名 max‖M−I‖≈6.8e-4、5名全请求 3.4e-3) "
          "⇒ 未训基线允许与 C/D 差一丝；空间两名同样只保证初值≈I(它们也能学) —— 与 G14 那条口径提示同源，不算接线错")
    else:
        # [G14 阳性对照·跑前写死] 族内参数化必须满足六件事, 违一条当场停(仪器闸):
        #   (1) 初值 gate=tanh(0)=0 ⇒ W 恰为 0 ⇒ 任意 multi-hot 的 M 逐位=I ⇒ "第0步逐位不变"实测(不是软恒等近似)
        #   (2) 16 张候选图每张都逐行 one-hot、行和恰 1，且 g=0 那张恰是单位阵(网格 m_0=1)
        #   (3) 族里含得住目标: 0.55 与 0.7 到最近网格的相对误差都要 ≤0.1 —— 过不了这条就别怪"学不到"
        #   (4) 梯度活性: gate 初值就有梯度(∵ cand−I≠0)；a 的梯度带 gate 因子 ⇒ 初值恰为 0，
        #       所以实测"门确定性置 0.1 后 a 的梯度>0"，用完逐位还原 ⇒ 两阶段都能被推动，且如实写出 a 要等门开
        #   (5) 可学标量数 == 240+15 == 255(与 D 臂自由表 3840 比是 1/15)
        #   (6) 报一条口径差: mix 臂**不再**由构造保证"空间两名 M=I"(它也能学门) ⇒ 只保证初值是 I
        with torch.no_grad():
            _I = torch.eye(_NB, device=DEVICE)
            _mh = torch.zeros(4, NT, device=DEVICE)
            for _i, _ids in enumerate(([0], [3], [0, 3], list(range(NT)))):
                for _k in _ids: _mh[_i, _k] = 1.0
            _d1 = float((gen._bin_mult_M(_mh) - _I).abs().max())
            _P = gen.mix_maps                                    # [16,nb,nb] 冻结
            _hot = int(((_P.abs().sum(-1) - 1.0).abs() > 1e-6).sum())
            _idg0 = float((_P[0] - _I).abs().max())
            _wsum = int(((_P.sum(-1) - 1.0).abs() > 1e-6).sum())
            _rel = [float(np.min(np.abs(MIX_GRID - t)) / t) for t in (0.55, 0.7)]
            _near = [int(np.argmin(np.abs(MIX_GRID - t))) for t in (0.55, 0.7)]
            _npar = int(gen.mixW.numel() + gen.mixGate.numel())
        _probe = (torch.arange(_NB * _NB, device=DEVICE, dtype=torch.float32) % 13).view(_NB, _NB)
        # 梯度泛函用"与偏差表逐位点乘一个非常数确定表"——不能用 abs().sum(): 凸组合的每元都为正 ⇒ dL/dw_g 对所有 g
        # 相同 ⇒ softmax 的梯度被对称性精确抵消成 0，那是探针的错不是参数化的错(本轮首跑就是这样误报"族内学不动")
        _lin = lambda: (gen._bin_gain_eff() * _probe).sum()
        with torch.no_grad():                                    # cand−I 的幅度: 确定性把门置 1 读一次再逐位还原
            _g0 = gen.mixGate.detach().clone()
            gen.mixGate.data.fill_(1.0)
            _amp = float(gen._bin_gain_eff().abs().max())
            gen.mixGate.data.copy_(_g0)
            _d0 = float(gen._bin_gain_eff().abs().max())
        _gG = torch.autograd.grad(_lin(), gen.mixGate)[0]
        _gA0 = torch.autograd.grad(_lin(), gen.mixW)[0]
        with torch.no_grad():
            gen.mixGate.data.fill_(0.1)
        _gA1 = torch.autograd.grad(_lin(), gen.mixW)[0]
        with torch.no_grad():
            gen.mixGate.data.copy_(_g0)
            _d6 = float(gen.mixGate.abs().max())
        w(f"[G14 mix自检] 初值 max‖M−I‖={_d1:.6f}(判据 0.000000) | 候选 one-hot 不合格行={_hot}/256 行和不符={_wsum} "
          f"P(m_0)=I maxΔ={_idg0:.6f} | 0.55/0.7 最近网格 g={_near} 相对误差={_rel[0]*100:.2f}%/{_rel[1]*100:.2f}%(判据 ≤10%) | "
          f"可学标量={_npar}(判据 255) | 梯度活性 ‖∂/∂gate‖={float(_gG.norm()):.4f} ‖∂/∂a‖(gate=0)={float(_gA0.norm()):.6f} "
          f"→(gate=0.1)={float(_gA1.norm()):.4f} | 族幅度‖cand−I‖max={_amp:.4f} 还原后 ‖W‖max={_d0:.6f} ‖gate‖max={_d6:.6f}")
        if (_d1 != 0.0 or _hot or _wsum or _idg0 != 0.0 or max(_rel) > 0.1 or _npar != 255 or _amp <= 0.0
                or float(_gG.norm()) <= 0.0 or float(_gA1.norm()) <= 0.0 or _d0 != 0.0 or _d6 != 0.0):
            w("[自检不符] 乘子族参数化没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)
        w("  [G14 口径提示] mix 臂不构造保证'空间两名 M=I'(它们也允许学门) ⇒ 只保证初值 I；"
          "若训完看到 翻转/平移 的门≠0，那是这条路在替别处补差，属可观测事实不是接线错误")
if int(args.ycbcr_head):
    # [参数对账] 联合色头应当只加 3×(16×256)+257=12,545 个可训参数(Y/Cb/Cr 三个头 + 1 个逐位门)。
    #  对不上就说明有东西被重复构造或没接进 opt —— 上一轮 PL 二级只差 4,098 个参数却改了结论, 所以这条要显式核。
    _exp_d = 3 * (args.n_bins * args.encoder_width) + args.encoder_width
    w(f"[参数对账] ycbcr_head={args.ycbcr_head} 预期新增 {_exp_d:,} | 基线(头关)应为 {n_params - _exp_d:,} | "
      f"span={args.chroma_span} aux权重={args.ycbcr_aux} | 分数口径仍 16-bin RGB argmax ⇒ 与 65.15/S1/S2 可比"
      + (" | ★head=2: 逐通道 base 分类器已从输出中移除(输出字母真的换了)" if int(args.ycbcr_head) >= 2 else ""))

# ── 纯显存约束 ──
if DEVICE.type == "cuda":
    _free_b, _tot_b = torch.cuda.mem_get_info()
    w(f"[显存] 专用显存: 空闲{_free_b/2**30:.1f}GB / 总{_tot_b/2**30:.1f}GB")
    _cap_gb = (args.vram_cap if args.vram_cap > 0 else
               float(os.environ.get("M2_VRAM_CAP_GB", "6.0")))  # [R20] 旗标优先, 0=旧环境变量口径
    # 静态大上限在桌面应用中途涨显存时会把本进程挤进共享显存(用户观察到的"泄露虚拟内存")
    _eff_cap = max(0.5, min(_cap_gb, _free_b / 2**30 - 1.0))
    torch.cuda.set_per_process_memory_fraction(min(1.0, _eff_cap / (_tot_b / 2**30)))
    w(f"[显存] 本进程分配上限 {_eff_cap:.1f}GB (超限响亮OOM, 不溢出共享显存)")
    if int(args.pure_learn_ops) and _eff_cap < 9.0 and args.batch_size >= 16:
        # 实测口径: 远端 S2(PL一级, bs16) 的 GPU峰值是 10.34GB。cap 低于它时 bs16 不是"慢",
        # 而是**每一批都 OOM 被跳过** ⇒ 整场空转。要么降 bs, 要么抬 cap, 不能两个都留原值。
        w(f"  [预检不足] PL 一级 bs16 实测峰值 10.34GB > 本次上限 {_eff_cap:.1f}GB ⇒ 大概率全程空转。"
          f" 用 --batch_size 8(峰值约减半) 或把 M2_VRAM_CAP_GB 抬到 ≥11")

params = list(visual_embed.parameters()) + list(gen.parameters()) + list(p7.parameters())
opt = torch.optim.Adam(params, lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)

_AUX, _AUX_ON = [], [False]
_RESID = []                       # [R30] 考试期间 (格中心L1, 残差L1, 样本数) 记账     # 考试期间的 (纯CE, 色度L1, 样本数) 记账; 默认关, 不影响训练路径
_YCB = []                       # 考试期间的联合色头记账: (ceY,ceCb,ceCr,越界比例,样本数)

def run_batch(rgb, degbins, target_bin, attr_p, attr_i, task=None, sigma=None, mfeat=None, prompt=None, tgt_cont=None):
    pe = visual_embed(rgb.reshape(rgb.shape[0] * NPATCH, N_IN)).reshape(rgb.shape[0], NPATCH, 256)
    _ap = gen.attr_proj(F.normalize(attr_p, dim=-1))
    if getattr(gen, 'attr_w', None) is not None: _ap = gen.attr_w * _ap     # [PL二级] 这条混合的常数1.0变可学
    patch_tokens = pe + _ap                                                 # E2: 局部属性进路由
    wo, sv, _, _, _ = p7.forward_batch(patch_tokens, patch_tokens, [NPATCH] * rgb.shape[0], [NPATCH] * rgb.shape[0], last_loss=0.0)
    if args.resid_head and args.resid_input:
        gen._resid_src = rgb.reshape(rgb.shape[0], NPATCH, N_IN, 1)   # [R31] 连续输入值(源位)
    logits = gen(degbins, sv, F.normalize(attr_i, dim=-1), wo, task, sigma, mfeat, prompt)   # [b,16,768,16]
    if args.soft_ce > 0:
        # [R45] ordinal soft-CE: 邻档分一点概率质量, 差1档的惩罚小于差8档。边界行归一化保持行和=1。
        _tb45 = target_bin.reshape(-1)
        _oh45 = F.one_hot(_tb45, args.n_bins).float()
        _nb45 = torch.zeros_like(_oh45)
        _nb45[:, :-1] += _oh45[:, 1:]
        _nb45[:, 1:] += _oh45[:, :-1]
        _tg45 = (1.0 - args.soft_ce) * _oh45 + (args.soft_ce / 2.0) * _nb45
        _tg45 = _tg45 / _tg45.sum(-1, keepdim=True)
        ce = -(_tg45 * F.log_softmax(logits.reshape(-1, args.n_bins), dim=-1)).sum(-1).mean()
    else:
        ce = F.cross_entropy(logits.reshape(-1, args.n_bins), target_bin.reshape(-1))
    loss = ce
    if args.resid_head and tgt_cont is not None and getattr(gen, '_resid_parts', None) is not None:
        # [R30] 连续解码 = 档中心 + 档内有界残差; argmax 不可导 ⇒ 梯度只走残差, CE 仍管选档。
        _rp30 = gen._resid_parts
        _bin30 = logits.argmax(-1).reshape(tgt_cont.shape)
        _ctr30 = (_bin30.to(_rp30.dtype) + 0.5) / args.n_bins
        _val30 = _ctr30 + (0.5 * torch.tanh(_rp30) / args.n_bins).reshape(tgt_cont.shape)
        _rl30 = F.l1_loss(_val30, tgt_cont)
        loss = loss + args.resid_loss * _rl30
        gen._resid_rowl1 = (_val30 - tgt_cont).abs().mean(-1)
        gen._resid_rowctr = (_ctr30 - tgt_cont).abs().mean(-1)
        if _AUX_ON[0]:
            _RESID.append((float(F.l1_loss(_ctr30, tgt_cont).item()), float(_rl30.item()), tgt_cont.shape[0], float(torch.tanh(_rp30).abs().mean().item())))
    if args.cont_loss > 0:
        # [F3] 全通道期望值回归: softmax→格中心 的连续期望 对 目标格中心 做 L1, 补 CE-over-bins 丢的序数/连续信息
        _nb2 = args.n_bins
        _ctr2 = (torch.arange(_nb2, device=logits.device, dtype=logits.dtype) + 0.5) / _nb2
        _exp2 = (F.softmax(logits, dim=-1) @ _ctr2).reshape(target_bin.shape)
        _gt2 = (target_bin.to(_exp2.dtype) + 0.5) / _nb2
        loss = loss + args.cont_loss * F.l1_loss(_exp2, _gt2)
    if args.color_aux > 0:
        # [色度监督] 期望值解码 → YCbCr, 只罚 Cb/Cr(亮度已由逐 bin CE 管)。
        # 目标值取自 target_bin 的格中心: run_batch 里的 `rgb` 是退化图(见 :247), 干净图只以 bin 索引存在。
        _nb = args.n_bins
        _ctr = (torch.arange(_nb, device=logits.device, dtype=logits.dtype) + 0.5) / _nb
        _exp = (F.softmax(logits, dim=-1) @ _ctr).reshape(target_bin.shape)          # [b,16,768] ∈[0,1]
        _gt = (target_bin.to(_exp.dtype) + 0.5) / _nb
        def _chroma(x):
            r, g, b = x[..., 0::3], x[..., 1::3], x[..., 2::3]     # 位序 = (y*16+x)*3 + c, c∈R,G,B
            return (0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                    0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b)
        cb_h, cr_h = _chroma(_exp); cb_t, cr_t = _chroma(_gt)
        aux = F.l1_loss(cb_h, cb_t) + F.l1_loss(cr_h, cr_t)
        loss = ce + args.color_aux * aux
        # 分开记账, 免污染旧 CE 口径。只在考试期间记: 训练步的值混进来会算错, 而且会无界增长
        if _AUX_ON[0]:
            _AUX.append((float(ce.item()), float(aux.item()), target_bin.shape[0]))
    _nbh = args.n_bins
    if args.ycbcr_head and getattr(gen, "_ycb_parts", None) is not None:
        ly, lcb, lcr = gen._ycb_parts
        y_t, cb_t, cr_t, _out = gen._ycb_targets(target_bin)
        # 色头目标是逐像素(...,256), 而 ly/lcb/lcr 是逐通道位(...,768) ⇒ 把每个像素的色度目标铺给它自己的 3 个通道位。
        # 这正是"联合"的含义: 一个像素的 R/G/B 三个槽必须给出同一个 Y/Cb/Cr, 联合解码才自洽。
        y_t = y_t.repeat_interleave(3, dim=-1); cb_t = cb_t.repeat_interleave(3, dim=-1)
        cr_t = cr_t.repeat_interleave(3, dim=-1)
        ce_y = F.cross_entropy(ly.reshape(-1, _nbh), y_t.reshape(-1))
        ce_cb = F.cross_entropy(lcb.reshape(-1, _nbh), cb_t.reshape(-1))
        ce_cr = F.cross_entropy(lcr.reshape(-1, _nbh), cr_t.reshape(-1))
        loss = loss + args.ycbcr_aux * (ce_y + ce_cb + ce_cr)
        if _AUX_ON[0]:
            _YCB.append((float(ce_y), float(ce_cb), float(ce_cr), _out, int(target_bin.shape[0]), float(ce.item())))
    elif _AUX_ON[0]:
        # [量程审计·不受开关门控] 越界比例只依赖目标图与 --chroma_span, 对照臂也必须打印它。
        #  (§6 教训14: 上一轮色度 L1 的打印写在 if color_aux>0 里 ⇒ 对照臂从没输出过这个数, 判决差点拿不到)
        _out0 = gen._ycb_targets(target_bin)[3]
        _YCB.append((0.0, 0.0, 0.0, _out0, int(target_bin.shape[0]), float(ce.item())))
    if args.ycbcr_head and getattr(gen, "_ycb_parts", None) is None and not getattr(gen, "_ycb_warned", False):
        gen._ycb_warned = True
        w("  [响亮警告] --ycbcr_head=1 但前向没产出 Y/Cb/Cr logits ⇒ 这条机制在当前路径在场但不生效"
          "(v2 路径未接, 只有 S/PL 路径接了)。当前分数不能当作「色度因子化输出头已判定」的证据。")
    if args.pixel_cont and tgt_cont is not None and getattr(gen, '_refined', None) is not None:
        loss = loss + args.pixel_loss * _pixel_struct_loss(gen._refined, tgt_cont)   # [R51]
    if args.refine_cnn and tgt_cont is not None:
        # [R49] 粗解码(规则后logits argmax→档中心) → CNN残差精炼 → 光滑RGB。治斑点噪声。
        _ctr49 = (torch.arange(args.n_bins, device=logits.device, dtype=logits.dtype) + 0.5) / args.n_bins
        _coarse49 = _ctr49[logits.argmax(-1).reshape(logits.shape[0], -1)]      # [b,12288]
        _cimg49 = _pm_to_img(_coarse49).permute(0, 3, 1, 2)                     # [b,3,64,64]
        _rimg49 = (_cimg49 + gen.refine_cnn(_cimg49)).clamp(0.0, 1.0)
        _ref49 = _img_to_pm(_rimg49.permute(0, 2, 3, 1))                        # [b,12288]
        gen._refined = _ref49
        loss = loss + args.refine_loss * F.l1_loss(_ref49, tgt_cont)
    with torch.no_grad():
        acc = (logits.argmax(-1) == target_bin).float().mean().item()
    return loss, acc, logits

# ════════════════════════ 考试 ════════════════════════
exam_rng = np.random.default_rng(SEED + 1)
def build_exam():
    return build_pairs(exam_labeled + (exam_generic if args.use_generic else []),
                       None, exam_rng, per_task=True)

def exam_eval(tag, save_png=False):
    """分块评估: 整批108对峰值~13GB会随桌面显存波动OOM(E5-E600事故), 分块降到训练同级"""
    rgb, degbins, bins, attr_p, attr_i, tasks, sigmas, mfeats, prompts, pvals, tc = exam_cache   # [R8] pvals=名×强度指示
    n = rgb.shape[0]
    chunk = args.exam_chunk        # [R20] 原写死 24; 默认不变 ⇒ 逐位回归不受影响
    loss_sum = acc_sum = 0.0
    _AUX.clear(); _AUX_ON[0] = True          # 只统计考试块, 训练步不入账
    pred_all = torch.empty(n, NPATCH * N_IN, dtype=torch.long, device=DEVICE)
    _rrow_c = torch.zeros(n, device=DEVICE); _rrow_v = torch.zeros(n, device=DEVICE)   # [R30]
    _src_acc = torch.zeros(n, device=DEVICE); _cp_acc = torch.zeros(n, device=DEVICE)  # [R34]
    _src_bin_t = (torch.tensor(np.stack(EXAM_SRC_BIN), dtype=torch.long, device=DEVICE)
                  if args.comp_probe and len(EXAM_SRC_BIN) == n else None)
    pred_exp_all = torch.empty(n, NPATCH * N_IN, dtype=torch.long, device=DEVICE)
    refined_all = torch.zeros(n, NPATCH * N_IN, dtype=torch.float32, device=DEVICE)   # [R49]
    _nb = args.n_bins
    _ctr_row = (torch.arange(_nb, device=DEVICE, dtype=torch.float32) + 0.5) / _nb
    with torch.no_grad():
        for i in range(0, n, chunk):
            sl = slice(i, min(n, i + chunk))
            if args.pure_learn_ops: gen.pl_probe = (i == 0)     # 只在第一块采一次, 免整场考试反复算
            loss, acc, logits = run_batch(rgb[sl], degbins[sl], bins[sl], attr_p[sl], attr_i[sl],
                                          tasks[sl], sigmas[sl], mfeats[sl], pvals[sl], tgt_cont=tc[sl])
            if (args.refine_cnn or args.pixel_cont) and getattr(gen, '_refined', None) is not None:
                refined_all[sl] = gen._refined
            if args.resid_head and getattr(gen, '_resid_rowl1', None) is not None:
                _rrow_c[sl] = gen._resid_rowctr; _rrow_v[sl] = gen._resid_rowl1
            if args.comp_probe and _src_bin_t is not None:
                with torch.no_grad():
                    _bs34 = sl.stop - sl.start
                    _pre34 = gen._pre_logits.reshape(_bs34, -1, _nb).argmax(-1)
                    _src_acc[sl] = (_pre34 == _src_bin_t[sl]).float().mean(-1)
                    _cp34 = gen.apply_rules(F.one_hot(degbins[sl], _nb).float(), pvals[sl])
                    _cp_acc[sl] = (_cp34.argmax(-1).reshape(_bs34, -1) == bins[sl].reshape(_bs34, -1)).float().mean(-1)
            bs_ = sl.stop - sl.start
            loss_sum += loss.item() * bs_; acc_sum += acc * bs_
            pred_all[sl] = logits.argmax(-1).reshape(bs_, -1)
            # 期望值解码: 量化器与 to_bins(:217) 逐字相同 floor(x*n_bins) 再钳位, 免引入第二种取整约定
            _e = (F.softmax(logits.float(), dim=-1) @ _ctr_row).reshape(bs_, -1)
            pred_exp_all[sl] = (_e * _nb).floor().clamp(0, _nb - 1).long()
    if args.pure_learn_ops: gen.pl_probe = False
    _AUX_ON[0] = False        # ★无条件复位。旧写法把这行写在 if color_aux>0 里面 ⇒ 对照臂考完试后标志永久为真;
                              #  过去没有副作用纯粹因为记账点也在同一个开关里(巧合), 但新的"不受开关门控"的量程审计
                              #  一进来就会变成每个训练步都往列表里堆 ⇒ 400 轮无界增长。这是"监控挂在开关上"的第二例。
    if _AUX:
        _n = sum(x[2] for x in _AUX)
        _ce = sum(x[0] * x[2] for x in _AUX) / _n
        _ax = sum(x[1] * x[2] for x in _AUX) / _n
        w(f"    [色度辅助{tag}] 纯CE={_ce:.4f}（这一项才与 A–E 的历史 CE 同口径可比） "
          f"| Cb+Cr 平均L1={_ax:.4f} | 权重={args.color_aux} | 合成={_ce + args.color_aux * _ax:.4f}")
        _AUX.clear()
    if args.resid_head and _RESID:
        _n30 = sum(x[2] for x in _RESID)
        _c30 = sum(x[0] * x[2] for x in _RESID) / _n30
        _v30 = sum(x[1] * x[2] for x in _RESID) / _n30
        _m30 = sum(x[3] * x[2] for x in _RESID) / _n30
        w(f"    [残差连续{tag}] 格中心L1={_c30:.5f} → 残差L1={_v30:.5f} (相对{(_c30-_v30)/max(_c30,1e-9)*100:+.2f}%) mean|tanh(resid)|={_m30:.4f}")
        _RESID.clear()
        if len(EXAM_ROW_KIND) == n:
            _k30 = {}
            for _i30, _kd30 in enumerate(EXAM_ROW_KIND): _k30.setdefault(_kd30, []).append(_i30)
            for _kd30 in sorted(_k30):
                if not _kd30.startswith("unseenCross"): continue
                _ix30 = torch.tensor(_k30[_kd30], device=DEVICE)
                _cc = float(_rrow_c[_ix30].mean()); _vv = float(_rrow_v[_ix30].mean())
                w(f"      [残差连续{tag}] {_kd30:<26} 中心={_cc:.5f} 残差={_vv:.5f} ({(_cc-_vv)/max(_cc,1e-9)*100:+.2f}%)")
    if args.comp_probe and _src_bin_t is not None and len(EXAM_ROW_KIND) == n:
        _act34 = (pred_all == bins.reshape(n, -1)).float().mean(-1)
        _k34 = {}
        for _i34, _kd34 in enumerate(EXAM_ROW_KIND): _k34.setdefault(_kd34, []).append(_i34)
        for _kd34 in sorted(_k34):
            if not _kd34.startswith(("unseenCross", "unseenChanMult", "single-", "unseen0")): continue
            _ix34 = torch.tensor(_k34[_kd34], device=DEVICE)
            w(f"      [探针三角{tag}] {_kd34:<28} 源重建={float(_src_acc[_ix34].mean())*100:6.2f}% "
              f"copy上限={float(_cp_acc[_ix34].mean())*100:6.2f}% 实际={float(_act34[_ix34].mean())*100:6.2f}%")
        _base40 = [i for i, k in enumerate(EXAM_ROW_KIND) if k == "base"]
        _ctr41 = (torch.arange(_nb, device=DEVICE, dtype=torch.float32) + 0.5) / _nb
        _src_cont_t = (torch.tensor(np.stack(EXAM_SRC_CONT), dtype=torch.float32, device=DEVICE)
                       if args.comp_probe and len(EXAM_SRC_CONT) == n else None)
        for _t40 in range(3):
            _ix40 = torch.tensor([i for i in _base40 if int(tasks[i]) == _t40], device=DEVICE)
            if _ix40.numel():
                _dev40 = (pred_all[_ix40].float() - bins.reshape(n, -1)[_ix40].float()).abs().mean()
                _l1ex = ""
                if _src_cont_t is not None:
                    _dec41 = _ctr41[pred_all[_ix40]]
                    _l1ex = f" 目标L1={float((_dec41 - tc[_ix40]).abs().mean()):.5f}"   # [R41修正] 用真连续目标(tc), 不是源图
                w(f"      [base分任务{tag}] task{_t40} acc={float(_act34[_ix40].mean())*100:6.2f}% 偏离={float(_dev40):.3f}档{_l1ex} (n={_ix40.numel()})")
        _t0ix = [i for i in _base40 if int(tasks[i]) == 0]
        for _sv46 in (0.1, 0.2, 0.3):
            _ix46 = torch.tensor([i for i in _t0ix if abs(float(sigmas[i]) - _sv46) < 1e-6], device=DEVICE)
            if _ix46.numel():
                _l146 = float((_ctr41[pred_all[_ix46]] - tc[_ix46]).abs().mean()) if _src_cont_t is not None else -1
                w(f"      [去噪分σ{tag}] σ={_sv46} acc={float(_act34[_ix46].mean())*100:6.2f}% 目标L1={_l146:.5f} (n={_ix46.numel()})")
        if _src_cont_t is not None and _base40:
            _ixb41 = torch.tensor(_base40, device=DEVICE)
            # [R41修正] 地板=源图自身量化误差(源bins vs 源连续), 与目标无关。
            #   旧版用 target bins 被 base-task2 行污染(那些行实际带 G12 _seen 组合, 目标≠源)。
            _floor41 = (_ctr41[_src_bin_t[_ixb41]] - _src_cont_t[_ixb41]).abs().mean()
            w(f"      [量化地板{tag}] 16档中心解码 vs 连续源 L1={float(_floor41):.5f} (源图自身, 表示层固有误差)")
    if _YCB:
        _n = sum(x[4] for x in _YCB)
        _cy = sum(x[0] * x[4] for x in _YCB) / _n
        _cb = sum(x[1] * x[4] for x in _YCB) / _n
        _cr = sum(x[2] * x[4] for x in _YCB) / _n
        _oo = sum(x[3] * x[4] for x in _YCB) / _n
        _pc = sum(x[5] * x[4] for x in _YCB) / _n
        w(f"    [联合色头{tag}] 越界目标={_oo*100:.2f}%(span={args.chroma_span}) "
          f"| 三路CE Y={_cy:.4f} Cb={_cb:.4f} Cr={_cr:.4f} 合计={_cy+_cb+_cr:.4f} "
          f"| 主CE(不含本项)={_pc:.4f} ⇒ 比值={(_cy+_cb+_cr)/max(_pc,1e-9):.2f} "
          f"{'(与主CE同阶, 这项真的在参与)' if 0.2 < (_cy+_cb+_cr)/max(_pc,1e-9) < 5 else '(量级不匹配: 要么被主CE淹没, 要么反过来绑架它)'}"
          + ("" if args.ycbcr_head else " | 本臂 head=0 ⇒ 三路CE未计算, 只审计量程"))
        _YCB.clear()
    if getattr(gen, "_ycb_gate", None):
        _gg = gen._ycb_gate
        w(f"      [联合路门{tag}] mode={_gg['mode']} G均值={_gg['g_mean']:.3f} "
          f"值域[{_gg['g_min']:.3f},{_gg['g_max']:.3f}] sd={_gg['g_sd']:.3f} "
          f"贴端比例(G<0.05或>1.95)={_gg['sat']*100:.1f}% | 联合logits幅度上限={_gg['jl_absmax']:.2f} (末块读数)"
          + ("  ⇒ 网络正在把这条路关到 0" if _gg['g_mean'] < 0.2 else "")
          + ("  ⇒ 门贴到上限 2, 说明量程又成了硬约束" if _gg['g_mean'] > 1.9 else ""))
    loss = loss_sum / n; acc = acc_sum / n
    logits = None
    bins_flat = bins.reshape(n, -1)
    exact = (pred_all == bins_flat).all(dim=1).float()                  # [n] 整图全对
    per_task = {}
    for t in range(3):
        m = tasks == t
        if m.any():
            per_task[TASKS[t]] = (float(((pred_all == bins_flat)[m]).float().mean()),
                                  int(m.sum()), float(exact[m].mean()))
    sig_line = ""
    if sigmas is not None:                                             # E4: 去噪按σ分档
        for sv_ in (0.10, 0.20, 0.30):
            m = (tasks == 0) & (sigmas == sv_)
            if m.any():
                sig_line += f" | σ{sv_:.1f}={float(((pred_all == bins_flat)[m]).float().mean())*100:.1f}%"
    w(f"  [考试{tag}] 总pixel_acc={acc*100:.2f}% | 整图全对={exact.mean()*100:.1f}% | CE={loss:.4f}{sig_line}")
    for tname, (a, n_, e) in per_task.items():
        w(f"    {tname}: pixel_acc={a*100:.2f}% ({n_}对) 整图全对={e*100:.1f}%")
    # [R6 尺子] 词表外强度坍缩: 同名单元形(训练乘子) vs 同名单元形(词表外乘子)。
    #   两类行的输入/attr/指示向量逐位相同(task=2 的 degrade 是确定性灰度、prompt=mhot([k])), 只有目标乘子不同
    #   ⇒ 差值就是"规则词表覆盖不到时, 后面模型能力还剩多少"。base 行单独报, 保住与 G5 原口径的可比性。
    #   只加打印, 不动 pred/loss/权重(与 exact_audit/cell_dev 同类)。
    if len(EXAM_ROW_KIND) == n and (args.exam_unseen_int or args.exam_singles or args.exam_unseen_shift):
        _pa6 = (pred_all == bins_flat).float()
        _dv6 = (pred_all - bins_flat).abs().float()
        _k6 = {}
        for _i6, _kd6 in enumerate(EXAM_ROW_KIND): _k6.setdefault(_kd6, []).append(_i6)
        def _r6(_kd):
            if _kd not in _k6: return None
            _ix = torch.tensor(_k6[_kd], device=_pa6.device)
            return (float(_pa6[_ix].mean()), float(exact[_ix].mean()),
                    float((_dv6[_ix] <= 1).float().mean()), float(_dv6[_ix].mean()), len(_k6[_kd]))
        def _p6(_kd):
            _v = _r6(_kd)
            if _v: w(f"      {_kd:<22} acc={_v[0]*100:6.2f}% 整图全对={_v[1]*100:5.1f}% ±1档={_v[2]*100:6.2f}%"
                     f" 平均偏离档数={_v[3]:.3f} (n={_v[4]})")
            return _v
        w(f"    [词表外强度{tag}] 分组按 EXAM_ROW_KIND 行类标签(不按行号猜: G11 按 i//8 猜栽过 4 格误配)")
        for _kd in ["base", "seen-int-combo"] + [f"single-{t}" for t in T_NAMES]: _p6(_kd)
        _dl6 = []
        for _m6 in [float(x) for x in str(args.unseen_mults).split(",") if x.strip()]:
            for _kn6 in (0, 1, 2):
                _kdu = f"unseen{_m6:.2f}-{T_NAMES[_kn6]}"
                _vu = _p6(_kdu); _vs = _r6(f"single-{T_NAMES[_kn6]}")
                if _vu is None: continue
                if _vs:
                    _d6 = (_vu[0] - _vs[0]) * 100; _dl6.append((_kdu, _d6))
                    w(f"        ↑同名已见强度={_vs[0]*100:.2f}% ⇒ 词表外Δ={_d6:+.2f}pp")
                else:
                    w("        (无同名 seen 对照: 要判坍缩必须同时开 --exam_singles 1)")
        _dl6s = []
        for _kd_u6 in [k for k in _k6 if k.startswith("unseenShift")]:
            _vu = _p6(_kd_u6); _vs = _r6(f"single-{T_NAMES[4]}")
            if _vu is None: continue
            if _vs:
                _d6s = (_vu[0] - _vs[0]) * 100; _dl6s.append((_kd_u6, _d6s))
                w(f"        ↑同名已见位移(8,8)={_vs[0]*100:.2f}% ⇒ 词表外Δ={_d6s:+.2f}pp")
        if _dl6s:
            _md6s = sum(x[1] for x in _dl6s) / len(_dl6s)
            w(f"      [位移坍缩量{tag}] 词表外位移 vs 已见位移(8,8): 平均Δ={_md6s:+.2f}pp ({len(_dl6s)}格, "
              f"最坏={min(_dl6s, key=lambda x: x[1])[0]} {min(x[1] for x in _dl6s):+.2f}pp)"
              + (" ⇒ 几何参数维仍是闭集: 名对但量不对就等于没说" if _md6s < -10 else
                 (" ⇒ 只部分外推" if _md6s < -2 else " ⇒ 几乎不坍缩(先确认目标真被位移改动)")))
        _kn6 = [k for k in _k6 if k.startswith("unseenKind")]
        if _kn6:
            _vs6 = _r6(f"single-{T_NAMES[3]}")       # 已见的"翻转"(水平镜像)单元形 = 同槽位的正确用法
            for _kd in sorted(_kn6):
                _vk = _r6(_kd)
                if _vk is None: continue
                w(f"      [词表外种类{tag}] {_kd:<34} acc={_vk[0]*100:6.2f}% ±1档={_vk[2]*100:6.2f}% "
                  f"平均偏离档数={_vk[3]:.3f} (n={_vk[4]})")
            if _vs6:
                w(f"        对照: 同槽位的正确用法 single-{T_NAMES[3]}(水平镜像)={_vs6[0]*100:.2f}% | "
                  f"目标本身(竖直翻转)与水平镜像是**不同的图** ⇒ 借槽那格分数越高越危险(说明它在自信地执行错的规则)")
        _ka6 = [k for k in _k6 if k.startswith("unseenKindAffine")]
        if _ka6:
            _vz = _r6(f"single-{T_NAMES[3]}")
            for _kd in sorted(_ka6):
                _vk = _r6(_kd)
                if _vk is None: continue
                _ex = ""
                if _vz:
                    _d14 = (_vk[0] - _vz[0]) * 100
                    _ex = f" | 对照 single-{T_NAMES[3]}={_vz[0]*100:.2f}% ⇒ Δ={_d14:+.2f}pp"
                w(f"      [种类参数化{tag}] {_kd:<30} acc={_vk[0]*100:6.2f}% ±1档={_vk[2]*100:6.2f}% "
                  f"平均偏离档数={_vk[3]:.3f} (n={_vk[4]}){_ex}")
            _nb14 = _r6("unseenKind-竖直翻转-借翻转槽")
            if _nb14 and _vz:
                w(f"        同目标同指示名、只差那 6 个系数: 无参数={_nb14[0]*100:.2f}% → 有参数见上 ⇒ "
                  f"差值就是'把种类从枚举改成参数'买到的东西")
        _cp6 = [k for k in _k6 if k.startswith("unseenChanPerm") and not k.endswith("零指示对照")]
        for _kd in sorted(_cp6):
            _vp = _r6(_kd); _vc = _r6(_kd + "-零指示对照")
            if _vp is None: continue
            _ex = f" | 同目标零指示对照={_vc[0]*100:.2f}% ⇒ 参数买到的 Δ={(_vp[0]-_vc[0])*100:+.2f}pp" if _vc else ""
            w(f"      [颜色种类参数化{tag}] {_kd:<26} acc={_vp[0]*100:6.2f}% ±1档={_vp[2]*100:6.2f}% "
              f"平均偏离档数={_vp[3]:.3f} (n={_vp[4]}){_ex}")
            if _vc and len(EXAM_ROW_NAMES) == n:
                # [R15 判别力分层] 整卷平均会把这一轴读钝: 实测 7 张考试图里 6 张是亮苹果(|R−B|均值 0.05~0.31),
                #   "完全不施加置换"就能拿到 64.04%(R↔B)/51.43%(轮转) 的档不变上限 ⇒ 只有高色反差的图才分得出。
                #   按图拆开报, 免得把"尺子钝"读成"机制无效"(§6 教训: 尺子本身也要被审)。
                _per = {}
                for _i15, _nm15 in enumerate(EXAM_ROW_NAMES):
                    _per.setdefault(_nm15, []).append(_i15)
                _parts = []
                for _nm15, _idxs in _per.items():
                    _a15 = [i for i in _idxs if EXAM_ROW_KIND[i] == _kd]
                    _b15 = [i for i in _idxs if EXAM_ROW_KIND[i] == _kd + "-零指示对照"]
                    if not _a15 or not _b15: continue
                    _pa = float(_pa6[torch.tensor(_a15, device=_pa6.device)].mean()) * 100
                    _pb = float(_pa6[torch.tensor(_b15, device=_pa6.device)].mean()) * 100
                    _parts.append(f"{_nm15.split('.')[0]} {_pb:.1f}→{_pa:.1f}({_pa-_pb:+.1f})")
                if _parts:
                    w(f"        [按图 零指示→带参数] " + " ".join(_parts))
        _cm6 = [k for k in _k6 if k.startswith("unseenChanMult") and k.endswith(("-零指示对照", "-借变暗名对照")) is False]
        for _kd in sorted(_cm6):
            _vp = _r6(_kd); _v0 = _r6(_kd + "-零指示对照"); _vb = _r6(_kd + "-借变暗名对照")
            if _vp is None: continue
            _ex = ""
            if _v0: _ex += f" | 零指示={_v0[0]*100:.2f}%(Δ={(_vp[0]-_v0[0])*100:+.2f}pp)"
            if _vb: _ex += f" | 借变暗名(错规则)={_vb[0]*100:.2f}%(Δ={(_vp[0]-_vb[0])*100:+.2f}pp)"
            w(f"      [颜色规则参数化{tag}] {_kd:<26} acc={_vp[0]*100:6.2f}% ±1档={_vp[2]*100:6.2f}% "
              f"平均偏离档数={_vp[3]:.3f} (n={_vp[4]}){_ex}")
        _x25 = [k for k in _k6 if k.startswith("unseenCross") and not k.endswith("-零指示对照")]
        for _kd in sorted(_x25):
            _vp = _r6(_kd); _v0 = _r6(_kd + "-零指示对照")
            if _vp is None: continue
            _ex = f" | 零指示={_v0[0]*100:.2f}%(Δ={(_vp[0]-_v0[0])*100:+.2f}pp)" if _v0 else ""
            w(f"      [族间组合{tag}] {_kd:<26} acc={_vp[0]*100:6.2f}% ±1档={_vp[2]*100:6.2f}% "
              f"平均偏离档数={_vp[3]:.3f} (n={_vp[4]}){_ex}")
        _sn6 = [_r6(f"single-{T_NAMES[_k7]}") for _k7 in (0, 1, 2)]
        _sn6 = [v[0] * 100 for v in _sn6 if v]
        if _sn6:
            # [R8 教训] 坍缩量单独看会被"地板效应"骗过(两边一起塌到地板 ⇒ Δ≈0 却什么都没学会) ⇒ 必须同栏报已见水平
            w(f"      [已见强度均值{tag}] 三个颜色名单元形 acc 均值={sum(_sn6)/len(_sn6):.2f}% (与坍缩量同栏判, 缺一不可)")
        if _dl6:
            _md6 = sum(x[1] for x in _dl6) / len(_dl6)
            w(f"      [坍缩量{tag}] 词表外强度 vs 同名已见强度: 平均Δ={_md6:+.2f}pp ({len(_dl6)}格, "
              f"最坏={min(_dl6, key=lambda x: x[1])[0]} {min(x[1] for x in _dl6):+.2f}pp, "
              f"最好={max(_dl6, key=lambda x: x[1])[0]} {max(x[1] for x in _dl6):+.2f}pp)"
              + (" ⇒ 命题成立: 规则覆盖不到, 后面能力就是死的" if _md6 < -10 else
                 (" ⇒ 只部分外推, 强度位仍主要靠词表" if _md6 < -2 else
                  " ⇒ 几乎不坍缩(可疑: 先查目标是否真被乘子改动, 见 [R6强度自检])")))
    # [G4 判据体检] 只加打印, 不动 pred/loss/权重 ⇒ 分数与加打印前逐位可比。
    #   对 per_task=True 的考卷, 行号 h = 图序号×3 + 任务号 ⇒ 能把 exact 归到具体哪张图。
    if args.exact_audit:
        _enames = [s["name"] for s in exam_labeled + (exam_generic if args.use_generic else [])]
        # [G8] 追加单元形后 行数≠图数×3 ⇒ 分组改按行名(EXAM_ROW_NAMES), 任务号直接读 tasks, 不再用 i//3 反推
        _rn = (EXAM_ROW_NAMES if len(EXAM_ROW_NAMES) == n else
               [(_enames[i // 3] if i // 3 < len(_enames) else "?") for i in range(n)])
        _pa = (pred_all == bins_flat).float()
        _grp = {}
        for _i, _nm in enumerate(_rn): _grp.setdefault(_nm, []).append(_i)
        w("    [逐图pixel_acc" + tag + "] " + " ".join(
            f"{_nm}={float(_pa[_v].mean())*100:.1f}%" for _nm, _v in _grp.items()))
        _hits = [int(h) for h in exact.nonzero(as_tuple=True)[0]]
        w("    [exact命中" + tag + "] " + (", ".join(f"{_rn[h]}/{TASKS[int(tasks[h])]}" for h in _hits)
                                           if _hits else f"无(n={n}对全为0)"))
    if args.prompt_cond:
        _rows = [int(i) for i in (tasks == 2).nonzero(as_tuple=True)[0]]
        # [R6] 词表外强度行的目标用的是非 MULT_TABLE 乘子, 而下面的"执行锚点"仍按 apply_combo(MULT_TABLE) 造
        #   ⇒ 那些行锚点与目标不同源, 混进 G5 汇总会把判据读坏。只从 G5 剔除, 它们的分数由 [词表外强度] 块单独报。
        if len(EXAM_ROW_KIND) == n:
            _rows = [i for i in _rows if not EXAM_ROW_KIND[i].startswith("unseen")]
        # [G5 组合泛化读数] 考试每个上色对挂一个"训练未见的二元组合"。判据跑前已在 g5_combo_probe.py 写死:
        #   执行锚点 = 把该组合施加到输入灰度上(完美执行变形、不知道色度)
        #   不执行锚点 = 完全不施加变形(等价于不读提示词)
        #   模型 corr ≥ 两锚点中点 ⇒ 执行;  ≤ 不执行锚点 ⇒ 没执行。pixel_acc 只报不判:
        #   上一轮已量出它对空间变形几乎不敏感(翻转 +6.6~+7.3pp、平移 -0.5~-0.7pp), 拿它当判据会误判。
        _pv = (pred_all.to(torch.float32) + 0.5) / _nb          # argmax 预测 → 格中心
        _tv = (bins.to(torch.float32) + 0.5) / _nb
        _gm_all = rgb.to(torch.float32)
        def _cor(a, b):
            a = a.reshape(-1).cpu().numpy(); b = b.reshape(-1).cpu().numpy()
            return float(np.corrcoef(a, b)[0, 1]) if (a.std() > 1e-9 and b.std() > 1e-9) else 0.0
        _agg, _verd = [], []
        for _r in _rows:
            _sel = prompts[_r].cpu().numpy()
            _tm = from_patch_major(_tv[_r].reshape(-1))
            _pm = from_patch_major(_pv[_r])
            _gm = from_patch_major(_gm_all[_r].reshape(-1))
            _orc = torch.tensor(apply_combo(_gm.cpu().numpy(), _sel), dtype=torch.float32, device=DEVICE)
            _cp, _co, _cio = _cor(_pm, _tm), _cor(_orc, _tm), _cor(_gm, _tm)
            _l1p = float((_pm - _tm).abs().mean()); _l1o = float((_orc - _tm).abs().mean())
            _l1i = float((_gm - _tm).abs().mean())
            _mid = 0.5 * (_co + _cio)
            _v = "执行" if _cp >= _mid else ("未执行" if _cp <= _cio else "中间/不可判")
            _verd.append(_v)
            _dv = (pred_all[_r] - bins_flat[_r]).abs().float()
            _dev = (f" 偏离档数: 0档={float((_dv == 0).float().mean())*100:.2f}% 1档={float((_dv == 1).float().mean())*100:.2f}%"
                    f" 2档={float((_dv == 2).float().mean())*100:.2f}% ≥3档={float((_dv >= 3).float().mean())*100:.2f}%"
                    f" 均值={float(_dv.mean()):.2f}" if args.cell_dev else "")
            w(f"      [{'单元形' if int(round(float(prompts[_r].sum()))) == 1 else '组合'} {combo_label(_sel)}] "
              f"pixel_acc={float((pred_all[_r] == bins_flat[_r]).float().mean())*100:.2f}% "
              f"整图全对={float(exact[_r])*100:.1f}% | corr 模型={_cp:.4f} 执行锚点={_co:.4f} 不执行锚点={_cio:.4f} "
              f"中点={_mid:.4f} ⇒ {_v} | L1 模型={_l1p:.4f} 执行锚点={_l1o:.4f} 不执行锚点={_l1i:.4f}{_dev}")
            _agg.append((_cp, _co, _cio, _l1p, _l1o, _l1i,
                         float((pred_all[_r] == bins_flat[_r]).float().mean())))
        if _agg:
            _a = np.array(_agg)
            w(f"    [G5汇总{tag}] {'上色全部格(组合7+单元形35)' if args.exam_singles else '未见组合'} {len(_rows)}对: "
              f"平均pixel_acc={_a[:,6].mean()*100:.2f}% "
              f"| corr 模型={_a[:,0].mean():.4f} 执行锚点={_a[:,1].mean():.4f} 不执行锚点={_a[:,2].mean():.4f} "
              f"| 中点判据={0.5*(_a[:,1].mean()+_a[:,2].mean()):.4f} | 判'执行'的格数={_verd.count('执行')}/{len(_verd)}"
              f" | L1 模型={_a[:,3].mean():.4f} 执行锚点={_a[:,4].mean():.4f} 不执行锚点={_a[:,5].mean():.4f}")
        if args.exam_singles and _agg:
            # [G8 判读·跑前写死的二分] 按"激活名数"分行: 1=单元形(元件会不会做), ≥2=组合(装配会不会做)。
            #   单元形只在**未见图**上判(已训图那格另报, 因为它是记忆): 每个变形在其 6 张未见图里数过线格数,
            #   ≥4/6 记"会做", ≤2/6 记"不会做", 3/6 记"未定"; `变暗` 两锚点同值(Pearson 对整体缩放不变)⇒ 天生判不动, 只报数。
            #   结论口径: 可判的 4 个单元形里 ≥3 个"会做"而组合仍不过 ⇒ 缺口在**组合装配**; ≤1 个"会做" ⇒ 缺口在**元件本身**。
            _rnG5 = (EXAM_ROW_NAMES if len(EXAM_ROW_NAMES) == n else ["?"] * n)
            _trnG5 = {s["name"] for s in (train_labeled[:args.max_sources] if args.max_sources else train_labeled)}
            _kinds = [int(round(float(prompts[_r].sum()))) for _r in _rows]
            _labs = [combo_label(prompts[_r].cpu().numpy()) for _r in _rows]
            _ok = []
            for _nm in sorted(set(l for k, l in zip(_kinds, _labs) if k == 1)):
                _ix = [i for i, (k, l) in enumerate(zip(_kinds, _labs)) if k == 1 and l == _nm
                       and _rnG5[_rows[i]] not in _trnG5]
                _ix_tr = [i for i, (k, l) in enumerate(zip(_kinds, _labs)) if k == 1 and l == _nm
                          and _rnG5[_rows[i]] in _trnG5]
                if not _ix: continue
                _b = np.array([_agg[i] for i in _ix])
                _p = sum(1 for i in _ix if _verd[i] == "执行")
                _unjudgeable = abs(_b[:, 1].mean() - _b[:, 2].mean()) < 1e-6
                _v = "不可判(两锚点同值)" if _unjudgeable else ("会做" if _p >= 4 else ("不会做" if _p <= 2 else "未定"))
                if not _unjudgeable: _ok.append(_p >= 4)
                _ptr = f" 已训图那格 corr={float(_agg[_ix_tr[0]][0]):.4f}" if _ix_tr else ""
                w(f"    [G8单元形·{_nm}] 未见{len(_ix)}图 过线{_p}/{len(_ix)} ⇒ {_v}"
                  f" | corr均值={_b[:,0].mean():.4f} 执行锚点={_b[:,1].mean():.4f} 不执行锚点={_b[:,2].mean():.4f}"
                  f" 中点线={0.5*(_b[:,1].mean()+_b[:,2].mean()):.4f}"
                  f" | L1={_b[:,3].mean()*100:.2f}% (锚点 {_b[:,4].mean()*100:.2f}%/{_b[:,5].mean()*100:.2f}%)"
                  f" | acc={_b[:,6].mean()*100:.2f}%{_ptr}")
            _cmb = [i for i, k in enumerate(_kinds) if k >= 2]
            _cp = sum(1 for i in _cmb if _verd[i] == "执行")
            _no = sum(_ok)
            _cc = ("组合装配" if _no >= 3 and _cp <= 3 else
                   ("元件本身" if _no <= 1 else
                    ("元件与组合都已做到" if _cp >= 5 else "混合(部分元件会做、组合仍不过)"))) if _ok else "元件本身"
            w(f"    [G8判读{tag}] 可判单元形 {_no}/{len(_ok)} 个[会做] | 组合过线 {_cp}/{len(_cmb)} "
              f"⇒ 缺口定位在 [{_cc}]")
    if args.g6_report:
        # [G6 验收·主人口径] 不只报 acc: 每张预测图都对**参考图(=该对的目标图)**算 平均 L1 与 皮尔逊相关。
        #   解码口径与 G5 相同(argmax 格中心 (bin+0.5)/n_bins); L1/corr 对逐位顺序不敏感, 故在 patch-major 上直算,
        #   与 from_patch_major 后的光栅序结果逐位相同(这点由 g6_png_audit.py 走 PNG 那条独立路核过)。
        _ctr = (torch.arange(_nb, device=DEVICE, dtype=torch.float32) + 0.5) / _nb
        _pv6 = _ctr[pred_all]                                                 # [n,12288] 预测格中心 ∈[0,1]
        _tv6 = _ctr[bins_flat]                                                  # 参考图
        _diff = (_pv6 - _tv6).abs()
        _enames6 = [s["name"] for s in exam_labeled + (exam_generic if args.use_generic else [])]
        # [G8] 行名对齐(追加单元形后 i//3 不成立); 无行名表时退回旧口径
        _rn6 = (EXAM_ROW_NAMES if len(EXAM_ROW_NAMES) == n else
                [(_enames6[i // 3] if i // 3 < len(_enames6) else "?") for i in range(n)])
        _trn6 = {s["name"] for s in (train_labeled[:args.max_sources] if args.max_sources else train_labeled)}
        def _cor6(a, b):
            a = a.double().cpu().numpy(); b = b.double().cpu().numpy()
            return float(np.corrcoef(a, b)[0, 1]) if (a.std() > 1e-12 and b.std() > 1e-12) else 0.0
        _l1_1 = _diff.mean(1); _cor_1 = torch.tensor([_cor6(_pv6[i], _tv6[i]) for i in range(n)], device=DEVICE)
        w(f"    [G6验收{tag}] 全体{n}对: 平均L1={_l1_1.mean()*100:.3f}% | 皮尔逊={_cor_1.mean():.4f} "
          f"| pixel_acc={acc*100:.2f}%")
        for t in range(3):
            m6 = tasks == t
            if m6.any():
                w(f"      {TASKS[t]}: L1={_l1_1[m6].mean()*100:.3f}% corr={_cor_1[m6].mean():.4f} "
                  f"pixel_acc={float(((pred_all == bins_flat)[m6]).float().mean())*100:.2f}% ({int(m6.sum())}对)")
        _seen = torch.tensor([_nm in _trn6 for _nm in _rn6], device=DEVICE)
        for _lb, _m6 in (("已训图", _seen), ("未见图", ~_seen)):
            if _m6.any():
                w(f"      [分层·{_lb}] {int(_m6.sum())}对: L1={_l1_1[_m6].mean()*100:.3f}% corr={_cor_1[_m6].mean():.4f} "
                  f"pixel_acc={float(((pred_all == bins_flat)[_m6]).float().mean())*100:.2f}%")
        _g6 = {}
        for _i, _nm in enumerate(_rn6): _g6.setdefault(_nm, []).append(_i)
        w("      [G6逐图] " + " ".join(f"{_nm}{'*' if _nm in _trn6 else ''}"
                                       f"(L1={float(_l1_1[_v].mean())*100:.2f}% c={float(_cor_1[_v].mean()):.3f})"
                                       for _nm, _v in _g6.items()))
    if args.decode_probe:
        # [尺子诊断] 三种解码口径 + 相邻档容错, 全部不改损失/权重, 因而与历史分数同存点可比。
        #   argmax = 现行分数; 期望值 = 用满整个分布后按 to_bins(:217) 同一个量化器取 bin;
        #   ±1 = |预测bin-目标bin|<=1 即"差一档"算对 —— 它把"颜色没学到"与"量化格子太粗"分开。
        _hit_a = (pred_all == bins_flat); _hit_e = (pred_exp_all == bins_flat)
        _dev_a = (pred_all - bins_flat).abs(); _dev_e = (pred_exp_all - bins_flat).abs()
        w(f"    [解码对照{tag}] argmax={_hit_a.float().mean()*100:.2f}% | 期望值={_hit_e.float().mean()*100:.2f}% "
          f"| Δ={(_hit_e.float().mean()-_hit_a.float().mean())*100:+.2f}pt | "
          f"±1档: argmax={(_dev_a<=1).float().mean()*100:.2f}% 期望值={(_dev_e<=1).float().mean()*100:.2f}% | "
          f"平均偏离档数: argmax={_dev_a.float().mean():.3f} 期望值={_dev_e.float().mean():.3f}")
        for t in range(3):
            m = tasks == t
            if m.any():
                _ha = _hit_a[m].float().mean(); _he = _hit_e[m].float().mean()
                _aa = (_dev_a[m] <= 1).float().mean(); _ae = (_dev_e[m] <= 1).float().mean()
                w(f"      {TASKS[t]}: argmax={_ha*100:.2f}% 期望值={_he*100:.2f}%(Δ{(_he-_ha)*100:+.2f}) "
                  f"| ±1: {_aa*100:.2f}%→{_ae*100:.2f}% | 偏离档数 {_dev_a[m].float().mean():.3f}→{_dev_e[m].float().mean():.3f}")
        # [格子占用] 上限若在"16 格太粗"，则应看到: 偏离集中在 1 档，且目标只占用少数格子(格子被浪费)。
        #  逐通道分开数: 位序 c = index%3 (c∈R,G,B)。Cb/Cr 是把目标格中心反算成 YCbCr 后再用同一量化器取 bin。
        def _occ(x, label):
            _bc = torch.bincount(x.reshape(-1), minlength=_nb).float()
            _p = _bc / _bc.sum().clamp(min=1)
            _ent = float(-(_p[_p > 0] * _p[_p > 0].log2()).sum())
            _used = int((_p > 0.01).sum())
            w(f"      {label}: 占用格数(>1%)={_used}/{_nb} 熵={_ent:.2f}/{np.log2(_nb):.2f} "
              f"分布={[round(float(v),3) for v in _p]}")
        _gt = (bins_flat.to(torch.float32).reshape(n, -1) + 0.5) / _nb            # 目标图的格中心 ∈[0,1]
        _r0, _g0, _b0 = _gt[:, 0::3], _gt[:, 1::3], _gt[:, 2::3]
        _cb_t = (0.5 - 0.168736 * _r0 - 0.331264 * _g0 + 0.5 * _b0)
        _cr_t = (0.5 + 0.5 * _r0 - 0.418688 * _g0 - 0.081312 * _b0)
        w(f"    [格子占用{tag}] 偏离档数分布: 0档={(_dev_a==0).float().mean()*100:.2f}% "
          f"1档={(_dev_a==1).float().mean()*100:.2f}% 2档={(_dev_a==2).float().mean()*100:.2f}% "
          f"≥3档={(_dev_a>=3).float().mean()*100:.2f}%")
        _occ(bins_flat.reshape(n, -1)[:, 0::3], "目标R")
        _occ((_cb_t * _nb).floor().clamp(0, _nb - 1).long(), "目标Cb")
        _occ((_cr_t * _nb).floor().clamp(0, _nb - 1).long(), "目标Cr")
        _occ(pred_all.reshape(n, -1)[:, 0::3], "预测R")
        w(f"    [量程扫描{tag}] {gen._ycb_sweep(bins)}")
        _pe = (pred_all.reshape(n, -1).to(torch.float32) + 0.5) / _nb              # argmax 预测的格中心
        _pr0, _pg0, _pb0 = _pe[:, 0::3], _pe[:, 1::3], _pe[:, 2::3]
        _cb_h = (0.5 - 0.168736 * _pr0 - 0.331264 * _pg0 + 0.5 * _pb0)
        _occ((_cb_h * _nb).floor().clamp(0, _nb - 1).long(), "预测Cb")
    if args.task_cond or args.prompt_cond:
        with torch.no_grad():
            tn = [round(float(gen.task_e.weight[i].norm()), 3) for i in range(3)] if args.task_cond else []
            sn = [round(float(gen.sigma_e.weight[i].norm()), 3) for i in range(4)] if args.task_cond else []
            mn = round(float(gen.mask_h.weight.norm()), 3) if args.task_cond else 0.0
            pn = [round(float(gen.prompt_e.weight[i].norm()), 3) for i in range(NT)] if args.prompt_cond else []
            pa = round(float(gen.prompt_attr.weight.norm()), 3) if args.prompt_cond else 0.0
        w(f"    [条件探针{tag}] 任务嵌入范数={tn} σ嵌入范数={sn} mask投影范数={mn}")
        if args.prompt_cond:
            w(f"    [G3提示词探针{tag}] ‖prompt_e‖({'/'.join(T_NAMES)})={pn} ‖prompt_attr‖={pa}"
              f" 增益(条件/attr)={float(2*torch.sigmoid(gen.cond_gain)):.3f}/{float(2*torch.sigmoid(gen.attr_gain)):.3f}")
        if args.refl_branch:
            with torch.no_grad():
                _al9 = torch.sigmoid(gen.perm_sel(prompts.float()))              # [n,2] = (镜像门, 平移门)
                _nm9 = prompts[:, 3] > 0.5; _nr9 = prompts[:, 4] > 0.5
                _dn = float(_al9[_nm9, 0].mean() - _al9[~_nm9, 0].mean()) if bool(_nm9.any()) and bool((~_nm9).any()) else float("nan")
                _dr = float(_al9[_nr9, 1].mean() - _al9[~_nr9, 1].mean()) if bool(_nr9.any()) and bool((~_nr9).any()) else float("nan")
                w(f"    [G7反射探针{tag}] ‖refl_lin‖={float(gen.refl_lin.weight.norm()):.3f} "
                  f"增益={float(2*torch.sigmoid(gen.refl_gain)):.3f} "
                  f"| [G9] ‖roll_lin‖={float(gen.roll_lin.weight.norm()):.3f} 增益={float(2*torch.sigmoid(gen.roll_gain)):.3f}")
                w(f"      [G9算子门{tag}] α(镜像) 请求该算子={float(_al9[_nm9,0].mean()):.3f} 未请求={float(_al9[~_nm9,0].mean()):.3f} Δ={_dn:+.3f}"
                  f" | α(平移) 请求={float(_al9[_nr9,1].mean()):.3f} 未请求={float(_al9[~_nr9,1].mean()):.3f} Δ={_dr:+.3f}"
                  f" (判据: Δ≥0.15 ⇒ 提示词在选算子; Δ<0.05 ⇒ 没选中)")
        if args.mult_compose:
            # [G12 读数] 把学到的档轴算子读回成"每通道隐含乘子": 行归一后取期望到档, 对原档中心做过原点最小二乘回归。
            #   理想置换图 ⇒ 斜率恰是该变形名的乘子(0.55/0.7); 组合那行是连乘之后的净斜率 ⇒ 与外部探针 g12 同一条尺的两个视图
            with torch.no_grad():
                _ct = (torch.arange(args.n_bins, device=DEVICE, dtype=torch.float32) + 0.5) / args.n_bins
                _den = float((_ct * _ct).sum())

                def _imp(ids):
                    M = gen._bin_mult_M(torch.tensor(mhot(ids), dtype=torch.float32, device=DEVICE).view(1, -1))
                    P = M / M.abs().sum(-1, keepdim=True).clamp(min=1e-6)
                    _to = (P * _ct).sum(-1)                                    # [1,3,nb] 期望到的档中心
                    return ([round(float((_to[0, c] * _ct).sum() / _den), 3) for c in range(3)],
                            round(float(P.max(-1).values.mean()), 3))
                _s3 = " ".join(f"{T_NAMES[k]}={'/'.join(map(str, _imp([k])[0]))}" for k in range(3))
                _c7 = " ".join(f"{combo_label(mhot(c))}={'/'.join(map(str, _imp(c)[0]))}" for c in EXAM_COMBOS)
                _pur = _imp([0])[1]
                w(f"    [G12乘子探针{tag}] ‖W_eff‖={float(gen._bin_gain_eff().norm()):.3f} 行峰度={_pur:.3f}")
                if args.mult_mix:
                    # [G14 读数] 族内参数化的三个可读量: 门(强度)、softmax 压在哪张候选上、净乘子 Σw·m(含"恒等候选还剩多少")
                    _wt = torch.softmax(MIX_BETA * gen.mixW, dim=-1)                  # [NT,3,16]
                    _gt = torch.tanh(gen.mixGate)                                     # [NT,3]
                    _gi = _wt.argmax(-1)                                              # [NT,3]
                    _netm = (_wt * torch.from_numpy(MIX_GRID).to(_wt.device)).sum(-1)  # [NT,3] 净乘子(混合期望)
                    _txt = " ".join(f"{T_NAMES[k]}[g={'/'.join(str(int(v)) for v in _gi[k])} "
                                    f"门={'/'.join(f'{float(v):+.3f}' for v in _gt[k])} "
                                    f"净乘子={'/'.join(f'{float(v):.3f}' for v in _netm[k])} "
                                    f"恒等权重={'/'.join(f'{float(v):.3f}' for v in _wt[k, :, 0])}]" for k in range(NT))
                    w(f"      [G14 mix读数{tag}] {_txt}")
                    w(f"        [G14 mix目标] 偏红/偏绿 应给受影响通道净乘子≈0.55(g=5 或 6)、变暗≈0.70(g=3)、"
                      f"翻转/平移 两名应留在门≈0(初值 0，非构造保证)；组合格的净乘子是两图连乘后的档位映射，不必等于标量之积")
                w(f"      [G12隐含乘子{tag}] 单元形[{_s3}] 组合[{_c7}] (目标: 偏红/偏绿=0.55, 变暗=0.7, 组合=两者之积)")
                if args.mult_convex:
                    # [G15 读数] 凸臂只有"压在哪张候选 + 恒等还剩多少 + 净乘子"三样可看；
                    #   另加**本张考卷实测**的非负性与行和(这才是"概率迁移"语义被守住的直接证据，不是构造上的声称)
                    _wt = torch.softmax(MIX_BETA * (gen.mixW + gen.mix_bias), dim=-1)          # [NT,3,16]
                    _gi = _wt.argmax(-1)
                    _netm = (_wt * torch.from_numpy(MIX_GRID).to(_wt.device)).sum(-1)           # [NT,3]
                    _Me = gen._bin_mult_M(prompts)                                             # [n,3,nb,nb] 本张考卷
                    _txt = " ".join(f"{T_NAMES[k]}[g={'/'.join(str(int(v)) for v in _gi[k])} "
                                    f"w0={'/'.join(f'{float(v):.3f}' for v in _wt[k, :, 0])} "
                                    f"净乘子={'/'.join(f'{float(v):.3f}' for v in _netm[k])}]" for k in range(NT))
                    w(f"      [G15 convex读数{tag}] {_txt}")
                    w(f"        [G15 凸性核账{tag}] 考卷 {prompts.shape[0]} 行的 M: 负元素={int((_Me < -1e-9).sum())} "
                      f"min元素={float(_Me.min()):.6f} 行和最大偏差={float((_Me.sum(-1) - 1.0).abs().max()):.6f} "
                      f"(判据: 0 / ≥0 / ≤1e-5 ⇒ 这一路始终是概率迁移，G14 的负权重逃逸口已关死)")
                if args.mult_oracle:
                    _dfrz = float((gen.bin_gain - oracle_bin_maps(args.n_bins).to(gen.bin_gain.device)).abs().max())
                    w(f"      [G13冻结核账{tag}] max‖bin_gain−理想图‖={_dfrz:.6f} "
                      f"requires_grad={bool(gen.bin_gain.requires_grad)} "
                      f"grad={'None' if gen.bin_gain.grad is None else round(float(gen.bin_gain.grad.norm()), 6)} "
                      f"(判据: 三个考试点都要 0.000000 / False / None ⇒ 这张图确实没被梯度碰过)")
    if args.pure_learn_ops and getattr(gen, '_pl_stats', None):
        s = gen._pl_stats
        w(f"    [PL探针{tag}] 14门均值={s['g_mean']}")
        w(f"      |G-1|均={s['g_dev']:.4f} 值域[{s['g_min']:.3f},{s['g_max']:.3f}] h_scale={s['h_scale']:.3f} | "
          f"核偏离初值(相对): avg3={s['d_a3']:.4f} avg5={s['d_a5']:.4f} shiftM={s['d_sM']:.4f}")
        w(f"      条件‖c‖={s['c_norm']:.3f} 身份‖h_in‖={s['hin_norm']:.3f} 比值={s['c_norm']/max(s['hin_norm'],1e-9):.3f}")
        if 'hin_raw_norm' in s:
            _inj = s['cond_gain'] * s['c_norm']                    # 注入向量范数, 分母是**已归一**的身份(‖u‖=1)
            w(f"      [G3注入量级] 归一前‖h_in‖={s['hin_raw_norm']:.1f} 旧口径比值={s['c_norm']/max(s['hin_raw_norm'],1e-9):.7f}"
              f" | 新口径 ‖增益×c‖/‖单位身份‖={_inj:.4f} (增益={s['cond_gain']:.3f} attr增益={s['attr_gain']:.3f} ‖prompt_e‖={s['prompt_c']:.3f})")
        if int(args.pure_learn_ops) >= 2:
            w(f"      [PL二级] 词典偏离初值={s['d_be']:.4f} 词典离初值正交基偏差={s['be_dev']:.4f} "
              f"attr_w={s['attr_w']:.4f}")
        _nm = ['自己像素','3x3','5x5','右移','左移','下移','上移','右下','左下','右上','左上','邻域注意力','P7上下文','身份注入']
        for _t, _k in enumerate(['去噪','遮补','上色']):
            _v = (s.get('per_task') or {}).get(_t)
            if _v:
                w(f"        [{_k}] " + " ".join(f"{_nm[i]}={_v[i]:.3f}" for i in range(14)))
    if save_png or args.save_png:                                       # [G6] 旧口径只有末轮传 True; 现在可用开关强开
        _ref49 = refined_all if (args.refine_cnn or args.pixel_cont) else None
        _dump_pngs(rgb, bins_flat, pred_all, tasks, max_n=int(args.dump_max), refined=_ref49)
        _i2i_metrics(rgb, pred_all, tc, tasks, refined=_ref49)          # [R48/R49] 真出图验收
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return float(acc)

def _dump_pngs(rgb, bins_flat, pred_flat, tasks, max_n=9, refined=None):
    shown = 0
    # [G6] 文件名带考卷图名 ⇒ 独立复核(g6_png_audit.py)按"图名+任务"配对, 不靠行序猜
    _en0 = [s["name"].rsplit(".", 1)[0] for s in exam_labeled + (exam_generic if args.use_generic else [])]
    # [G8] 行名对齐(EXAM_ROW_NAMES), 追加单元形后仍指向正确的图
    _en = ([s.rsplit(".", 1)[0] for s in EXAM_ROW_NAMES] if len(EXAM_ROW_NAMES) == rgb.shape[0] else
           [_en0[i // 3] if i // 3 < len(_en0) else 'x' for i in range(rgb.shape[0])])
    for i in range(rgb.shape[0]):
        if shown >= max_n: break
        inp = from_patch_major(rgb[i].cpu())                        # patch-major → [64,64,3]
        tgt = from_patch_major(((bins_flat[i].float() + 0.5) / args.n_bins).cpu())
        prd = (from_patch_major(refined[i].cpu()) if refined is not None
               else from_patch_major(((pred_flat[i].float() + 0.5) / args.n_bins).cpu()))
        if args.decode_smooth:
            prd = torch.from_numpy(_smooth_img(prd.numpy()))
        rows = np.concatenate([inp.numpy(), prd.numpy(), tgt.numpy()], axis=1)
        big = cv2.resize((rows * 255).round().astype(np.uint8), (192 * 8, IMG * 8), interpolation=cv2.INTER_NEAREST)
        imwrite_u8(os.path.join(OUT_DIR, f"m2_exam_{_en[i]}_T{tasks[i].item()}_{i:02d}_in-pred-tgt.png"),
                   cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        shown += 1
    w(f"  [png] {shown}张三联图(64×64, ×8放大) → {OUT_DIR}")

def _i2i_metrics(rgb, pred_flat, tc, tasks, refined=None):
    """[R48] 主人 i2i 验收口径: 预测连续RGB vs 参考连续RGB 的平均L1 + 皮尔逊相关, 按任务/行类聚合。
       预测解码=档中心(argmax bin), 与三联图同口径; 参考=tc(真连续目标)。"""
    n = rgb.shape[0]
    ctr = (torch.arange(args.n_bins, device=DEVICE, dtype=torch.float32) + 0.5) / args.n_bins
    pred_c = refined if refined is not None else ctr[pred_flat]   # [n,12288] 连续预测(R49用精炼)
    if args.decode_smooth:
        _imgs = _pm_to_img(pred_c).cpu().numpy()
        _sm = np.stack([_smooth_img(_imgs[i]) for i in range(_imgs.shape[0])])
        pred_c = _img_to_pm(torch.from_numpy(_sm).to(pred_c.device))   # [R50] 转回patch-major再比
    l1_row = (pred_c - tc).abs().mean(-1)                     # [n]
    pc = pred_c.cpu().numpy(); tg = tc.cpu().numpy()
    pear = np.array([np.corrcoef(pc[i], tg[i])[0, 1] for i in range(n)])
    def _agg(ix, label):
        ix = np.array(ix)
        if len(ix) == 0: return
        w(f"      [i2i验收 {label}] L1={float(l1_row[ix].mean()):.4f} 皮尔逊={float(np.nanmean(pear[ix])):.4f} (n={len(ix)})")
    w(f"    [i2i验收 总体] L1={float(l1_row.mean()):.4f} 皮尔逊={float(np.nanmean(pear)):.4f}")
    for t in range(3):
        _agg([i for i in range(n) if int(tasks[i]) == t], f"task{t}")
    if len(EXAM_ROW_KIND) == n:
        kinds = {}
        for i, k in enumerate(EXAM_ROW_KIND): kinds.setdefault(k, []).append(i)
        for k in sorted(kinds):
            if k.startswith(("single-", "unseenCross-", "unseenChanMult-")) and "对照" not in k:
                _agg(kinds[k], k)
    # 落盘逐行指标 csv, 供独立复核
    import csv as _csv
    with open(os.path.join(OUT_DIR, "i2i_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        wr = _csv.writer(f); wr.writerow(["row", "name", "kind", "task", "L1", "pearson"])
        for i in range(n):
            wr.writerow([i, EXAM_ROW_NAMES[i] if len(EXAM_ROW_NAMES)==n else "",
                         EXAM_ROW_KIND[i] if len(EXAM_ROW_KIND)==n else "", int(tasks[i]),
                         f"{float(l1_row[i]):.5f}", f"{float(pear[i]):.5f}"])

# ════════════════════════ 训练 ════════════════════════
rng_train = np.random.default_rng(SEED * 1000)
def build_train():
    _srcs = train_labeled[:args.max_sources] if args.max_sources else train_labeled  # 诊断: 4图记忆测试
    rgb_l, dgb_l, bin_l, ap_l, ai_l, task_l, sg_l, mf_l, pr_l, pv_l, tc_l = build_pairs(_srcs, args.k_labeled, rng_train)
    if not args.use_generic:
        return rgb_l, dgb_l, bin_l, ap_l, ai_l, task_l, sg_l, mf_l, pr_l, pv_l, tc_l
    rgb_g, dgb_g, bin_g, ap_g, ai_g, task_g, sg_g, mf_g, pr_g, pv_g, tc_g = build_pairs(train_generic, args.k_generic, rng_train)
    return (torch.cat([rgb_l, rgb_g]), torch.cat([dgb_l, dgb_g]), torch.cat([bin_l, bin_g]),
            torch.cat([ap_l, ap_g]), torch.cat([ai_l, ai_g]), torch.cat([task_l, task_g]),
            torch.cat([sg_l, sg_g]), torch.cat([mf_l, mf_g]), torch.cat([pr_l, pr_g]),
            torch.cat([pv_l, pv_g]), torch.cat([tc_l, tc_g]))

if args.spat_map:
    # [R11 阳性对照·跑前写死] 构造式空间算子必须由构造保证四件事, 违一条当场停(仪器闸, 不是判负闸):
    #   (1) 没请求任何空间名 ⇒ logits 逐位不变  (2) 镜像是对合: 连做两次回到原样(逐位)
    #   (3) 只请求平移 ⇒ 输出逐位等于按 roll_idx 取的那一份(接线没串到别的表上)
    #   (4) 同时请求镜像+平移 ⇒ 等于"先镜像后平移"(与 apply_combo 的顺序一致), 反序必须不同(否则说明顺序没被真的固定)
    with torch.no_grad():
        _x11 = torch.arange(2 * NPATCH * N_IN * 3, device=DEVICE, dtype=torch.float32)
        _x11 = (_x11 % 977).view(2, NPATCH, N_IN, 3)
        def _sp(pairs):
            v = torch.zeros(2, PV_W, device=DEVICE)
            for _b11, _k11 in pairs: v[_b11, _k11] = 1.0
            return gen._apply_spat_map(_x11, v)
        _e1 = float((_sp([]) - _x11).abs().max())
        _mm = torch.zeros(2, PV_W, device=DEVICE); _mm[:, 3] = 1.0
        _e2 = float((gen._apply_spat_map(gen._apply_spat_map(_x11, _mm), _mm) - _x11).abs().max())
        _ro = _x11.reshape(2, NPATCH * N_IN, 3).index_select(1, gen.roll_idx).reshape(2, NPATCH, N_IN, 3)
        _e3 = float((_sp([(0, 4), (1, 4)]) - _ro).abs().max())
        _both = _sp([(0, 3), (0, 4), (1, 3), (1, 4)])
        _mf = _x11.reshape(2, NPATCH * N_IN, 3).index_select(1, gen.refl_idx).reshape(2, NPATCH, N_IN, 3)
        _fr = _mf.reshape(2, NPATCH * N_IN, 3).index_select(1, gen.roll_idx).reshape(2, NPATCH, N_IN, 3)
        _e4 = float((_both - _fr).abs().max())
        _rf = _ro.reshape(2, NPATCH * N_IN, 3).index_select(1, gen.refl_idx).reshape(2, NPATCH, N_IN, 3)
        _e5 = float((_both - _rf).abs().max())
        w(f"[R11构造自检] 无请求逐位不变={_e1:.3e} | 镜像对合回到原样={_e2:.3e} | 单平移==roll_idx取={_e3:.3e} "
          f"| 镜像+平移==先镜像后平移={_e4:.3e} | 与反序(先平移后镜像)之差={_e5:.3e} "
          f"(判据: 前四项须 ≤1e-6 且第五项 >0 ⇒ 顺序真被固定住, 不是两条路恰好等价)")
        if max(_e1, _e2, _e3, _e4) > 1e-6 or _e5 <= 0.0:
            w("[自检不符] 构造式空间算子没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)

if args.roll_param:
    # [R12 阳性对照·跑前写死] 位移参数化必须由构造保证三件事, 违一条当场停(仪器闸, 不是判负闸):
    #   (1) 向量化现造表在 (8,8) 上与老的 Python 逐位循环表 _roll_index() **逐位相同**(两条独立实现)
    #   (2) 在 (4,4)/(8,0)/(12,12) 上, 现造表与 numpy 的 np.roll(经 apply_combo_m) **逐位相同**
    #   (3) _apply_spat_map 收到显式位移参数时, 输出逐位等于按现造表 gather 的那一份(接线没串)
    with torch.no_grad():
        _dv1 = int((gen._roll_tbl_for(8, 8, DEVICE) != _roll_tbl.to(DEVICE)).sum())
        _src12 = sources[0]["rgb"]
        _flat12 = to_patch_major(_src12).reshape(-1)
        _dv2 = {}
        for _sh12 in ((4, 4), (8, 0), (12, 12)):
            _np12 = to_patch_major(apply_combo_m(_src12, mhot([4]), None, _sh12)).reshape(-1)
            _gb12 = _flat12[gen._roll_tbl_for(_sh12[0], _sh12[1], torch.device("cpu")).numpy()]
            _dv2[str(_sh12)] = float(np.abs(_np12.astype(np.float64) - _gb12.astype(np.float64)).max())
        _x12 = torch.arange(NPATCH * N_IN * 2, device=DEVICE, dtype=torch.float32)
        _x12 = (_x12 % 977).view(2, NPATCH, N_IN, 1)
        _v12 = torch.zeros(2, NT + 2, device=DEVICE)
        _v12[:, 4] = 1.0
        _v12[0, NT] = 4.0 / IMG; _v12[0, NT + 1] = 4.0 / IMG        # 第0行请求 (4,4)
        _v12[1, NT] = 12.0 / IMG; _v12[1, NT + 1] = 12.0 / IMG      # 第1行请求 (12,12)
        _o12 = gen._apply_spat_map(_x12, _v12).reshape(2, NPATCH * N_IN, 1)
        _w12 = torch.stack([_x12[i].reshape(NPATCH * N_IN, 1).index_select(
            0, gen._roll_tbl_for(*sh, DEVICE)) for i, sh in enumerate(((4, 4), (12, 12)))])
        _dv3 = float((_o12 - _w12).abs().max())
        w(f"[R12构造自检] 现造表(8,8) vs 老表 不符位数={_dv1}/{NPATCH*N_IN} | 现造表 vs numpy.roll 最大逐位差={_dv2} "
          f"| _apply_spat_map 按行取表 maxΔ={_dv3:.3e} (判据: 第一项=0、第二项全 0.000000、第三项 ≤1e-6)")
        if _dv1 or max(_dv2.values()) != 0.0 or _dv3 > 1e-6:
            w("[自检不符] 位移参数化没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)

if args.affine_map:
    # [R14 阳性对照·跑前写死] 一般坐标映射必须由构造保证四件事, 违一条当场停(仪器闸, 不是判负闸):
    #   (1) 仿射参数取水平镜像 ⇒ 现造表与 G7 已核过的 refl_idx **逐位相同**(等价性: 新族必须包含老算子)
    #   (2) 仿射参数取 roll(8,8) ⇒ 现造表与 G9 已核过的 roll_idx **逐位相同**
    #   (3) 竖直翻转/旋转90 ⇒ 现造表与 numpy 的 apply_affine(独立实现)在真图上**逐位相同**
    #   (4) 指示向量不带仿射参数 ⇒ _apply_spat_map 输出与带参数路互不干扰(仿射行不会被旧表作用两次)
    with torch.no_grad():
        _af_hf = (1.0, 0.0, 0.0, -1.0, 0.0, float(IMG - 1))       # 水平镜像(像素单位)
        _af_rl = (1.0, 0.0, 0.0, 1.0, -8.0, -8.0)                 # roll(8,8)
        _e1 = int((gen._affine_tbl_for(_af_hf, DEVICE) != gen.refl_idx).sum())
        _e2 = int((gen._affine_tbl_for(_af_rl, DEVICE) != gen.roll_idx).sum())
        _src14 = sources[0]["rgb"]
        _flat14 = to_patch_major(_src14).reshape(-1)
        _e3 = {}
        for _nm14, _af14 in (("竖直翻转", (-1.0, 0.0, 0.0, 1.0, float(IMG - 1), 0.0)),
                             ("旋转90", (0.0, 1.0, -1.0, 0.0, 0.0, float(IMG - 1)))):
            _np14 = to_patch_major(apply_affine(_src14, _af14)).reshape(-1)
            _gb14 = _flat14[gen._affine_tbl_for(_af14, torch.device("cpu")).numpy()]
            _e3[_nm14] = float(np.abs(_np14.astype(np.float64) - _gb14.astype(np.float64)).max())
        _x14 = torch.arange(NPATCH * N_IN * 2, device=DEVICE, dtype=torch.float32)
        _x14 = (_x14 % 977).view(2, NPATCH, N_IN, 1)
        _v14 = torch.zeros(2, PV_W, device=DEVICE); _v14[:, 3] = 1.0            # 两行都借"翻转"名槽
        _v14[1, NT + 2:NT + 8] = torch.tensor((-1.0, 0.0, 0.0, 1.0, (IMG - 1) / IMG, 0.0), device=DEVICE)
        _o14 = gen._apply_spat_map(_x14, _v14).reshape(2, NPATCH * N_IN, 1)
        _r0 = _x14[0].reshape(NPATCH * N_IN, 1).index_select(0, gen.refl_idx)     # 第0行无仿射 ⇒ 走旧镜像表
        _r1 = _x14[1].reshape(NPATCH * N_IN, 1).index_select(
            0, gen._affine_tbl_for((-1.0, 0.0, 0.0, 1.0, float(IMG - 1), 0.0), DEVICE))
        _e4 = float((_o14 - torch.stack([_r0, _r1])).abs().max())
        w(f"[R14构造自检] 仿射==水平镜像 不符位={_e1}/{NPATCH*N_IN} | 仿射==roll(8,8) 不符位={_e2}/{NPATCH*N_IN} "
          f"| 现造表 vs numpy.apply_affine 最大逐位差={_e3} | 仿射行与旧路互不干扰 maxΔ={_e4:.3e} "
          f"(判据: 前两项=0、第三项全 0.000000、第四项 ≤1e-6)")
        if _e1 or _e2 or max(_e3.values()) != 0.0 or _e4 > 1e-6:
            w("[自检不符] 一般坐标映射没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)

if args.chan_perm:
    # [R15 阳性对照·跑前写死] 通道置换族必须由构造保证四件事, 违一条当场停(仪器闸, 不是判负闸):
    #   (1) 恒等置换 (0,1,2) ⇒ 现造表逐位等于 arange(即不动)
    #   (2) R↔B (2,1,0) ⇒ 现造表是对合(连做两次回到 arange)
    #   (3) R↔B / 轮转 ⇒ 现造表与 numpy 的 apply_chan_perm(独立实现)在真图上逐位相同
    #   (4) 未请求(参数全零) ⇒ _apply_chan_map 逐位原样返回
    with torch.no_grad():
        _ar = torch.arange(NPATCH * N_IN, device=DEVICE)
        _f1 = int((gen._chan_tbl_for((0, 1, 2), DEVICE) != _ar).sum())
        _t2 = gen._chan_tbl_for((2, 1, 0), DEVICE)
        _f2 = int((_t2[_t2] != _ar).sum())
        _src15 = sources[0]["rgb"]
        _flat15 = to_patch_major(_src15).reshape(-1)
        _f3 = {}
        for _nm15, _pm15 in (("R↔B", (2, 1, 0)), ("轮转RGB", (1, 2, 0))):
            _np15 = to_patch_major(apply_chan_perm(_src15, _pm15)).reshape(-1)
            _gb15 = _flat15[gen._chan_tbl_for(_pm15, torch.device("cpu")).numpy()]
            _f3[_nm15] = float(np.abs(_np15.astype(np.float64) - _gb15.astype(np.float64)).max())
        _x15 = torch.arange(NPATCH * N_IN * 2, device=DEVICE, dtype=torch.float32)
        _x15 = (_x15 % 977).view(2, NPATCH, N_IN, 1)
        _v15 = torch.zeros(2, PV_W, device=DEVICE)
        _f4 = float((gen._apply_chan_map(_x15, _v15) - _x15).abs().max())
        # [R15补] 光核"表对不对"不够, 还要核"表真的被作用到 logits 上"(R14 有这一条、R15 原先漏了):
        #   两行都请求 R↔B, 但第0行给参数、第1行不给 ⇒ 第0行必须等于手工 gather, 第1行必须原样。
        _v15b = torch.zeros(2, PV_W, device=DEVICE)
        _v15b[0, NT + 8:NT + 11] = torch.tensor((1.0, 0.5, 0.0), device=DEVICE)     # π=(2,1,0)
        _o15 = gen._apply_chan_map(_x15, _v15b).reshape(2, NPATCH * N_IN, 1)
        _exp0 = _x15[0].reshape(NPATCH * N_IN, 1).index_select(0, gen._chan_tbl_for((2, 1, 0), DEVICE))
        _f5 = max(float((_o15[0] - _exp0).abs().max()), float((_o15[1] - _x15[1].reshape(NPATCH * N_IN, 1)).abs().max()))
        w(f"[R15构造自检] 恒等置换不符位={_f1}/{NPATCH*N_IN} | R↔B 对合不符位={_f2}/{NPATCH*N_IN} "
          f"| 现造表 vs numpy.apply_chan_perm 最大逐位差={_f3} | 未请求逐位不变={_f4:.3e} "
          f"| 请求行==手工gather 且 未请求行原样 maxΔ={_f5:.3e} "
          f"(判据: 前两项=0、第三项全 0.000000、第四五项=0)")
        if _f1 or _f2 or max(_f3.values()) != 0.0 or _f4 != 0.0 or _f5 != 0.0:
            w("[自检不符] 通道置换族没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)

if args.chan_mult:
    # [R16 阳性对照·跑前写死] 逐通道乘子族必须由构造保证四件事, 违一条当场停(仪器闸, 不是判负闸):
    #   (1) 恒等乘子 (1,1,1) ⇒ M 逐位=I
    #   (2) **新族包含老预设**: 给 (1,0.55,0.55) 且不给名 ⇒ M 与"请求偏红、强度0.55"逐位相同(与 R14 仿射包含镜像同口径)
    #   (3) 三个词表外乘子组 ⇒ M 逐行和=1、元素非负(概率迁移, 不许出现负权重)
    #   (4) 期望档位移 vs to_bins(m_c·ctr) 偏离>1档的格数(插值固有价, 只报不判)
    with torch.no_grad():
        _I16 = torch.eye(_NB, device=DEVICE)
        def _cm16(vals):
            v = torch.zeros(1, PV_W, device=DEVICE)
            v[0, NT + 11:NT + 14] = torch.tensor(vals, device=DEVICE, dtype=torch.float32)
            return gen._int_map_M(v)[0]                                  # [3,nb,nb]
        _h1 = float((_cm16((1.0, 1.0, 1.0)) - _I16).abs().max())
        _pv16 = torch.zeros(1, PV_W, device=DEVICE); _pv16[0, 0] = 0.55     # 请求"偏红"、强度 0.55
        _named = gen._int_map_M(_pv16)[0]
        _h2 = float((_cm16((1.0, 0.55, 0.55)) - _named).abs().max())
        _CMS16 = [(1.00, 0.85, 0.70), (0.80, 0.90, 1.00), (0.90, 0.60, 0.90)]
        _h3 = 0.0
        for _c16 in _CMS16:
            _M16 = _cm16(_c16)
            _h3 = max(_h3, float((_M16.sum(-1) - 1.0).abs().max()), float((-_M16).clamp(min=0).max()))
        _ctr16 = (torch.arange(_NB, device=DEVICE, dtype=torch.float32) + 0.5) / _NB
        _bad16 = 0; _mx16 = 0.0
        for _c16 in _CMS16:
            _M16 = _cm16(_c16)
            _e16 = (_M16 * torch.arange(_NB, device=DEVICE, dtype=torch.float32)).sum(-1)     # [3,nb]
            _t16 = torch.stack([torch.clamp((float(_c16[_c]) * _ctr16 * _NB).floor(), 0, _NB - 1) for _c in range(3)])
            _d16 = (_e16 - _t16).abs()
            _bad16 += int((_d16 > 1.0).sum()); _mx16 = max(_mx16, float(_d16.max()))
        w(f"[R16构造自检] 恒等乘子 max‖M−I‖={_h1:.3e} | (1,.55,.55)无名 vs 请求偏红0.55 最大差={_h2:.3e} "
          f"| 三组词表外乘子 行和偏差/负权重={_h3:.3e} | 插值代价: 偏离>1档格数={_bad16}/144 最大偏离={_mx16:.2f}档 "
          f"(判据: 前两项 ≤1e-6、第三项 ≤1e-6; 插值代价只报不判)")
        if _h1 > 1e-6 or _h2 > 1e-6 or _h3 > 1e-6:
            w("[自检不符] 逐通道乘子族没按声明接线 ⇒ 停在这里, 不开训"); sys.exit(3)

w("\n[造对] 首轮训练对生成中...")
exam_cache = build_exam()          # 考试集: 全σ档固定参数
if args.exam_manifest:
    _mf = os.path.join(OUT_DIR, "exam_manifest.csv")
    _tk, _pr = exam_cache[5], exam_cache[8]
    with open(_mf, "w", encoding="utf-8", newline="") as _fh:
        _fh.write("row,image,task,sel_ids,label\n")
        for _i in range(len(EXAM_ROW_NAMES)):
            _s = [k for k in range(NT) if float(_pr[_i, k]) > 0.5]
            _fh.write(f"{_i},{EXAM_ROW_NAMES[_i]},{int(_tk[_i])},{'|'.join(map(str, _s))},"
                      f"{combo_label(_pr[_i].cpu().numpy())}\n")
    w(f"[考卷行表] {len(EXAM_ROW_NAMES)} 行 → {_mf}")

if args.exam_only:                 # [只评测] 载入存点读门/读数, 不建训练集不进循环
    _sd = torch.load(args.exam_only, map_location="cpu", weights_only=True)
    visual_embed.load_state_dict(_sd["visual_embed"])
    _missing = []
    try:
        gen.load_state_dict(_sd["gen"])                    # 严格: 结构对不上就先炸, 不许静默
    except RuntimeError as _e:
        _r = gen.load_state_dict(_sd["gen"], strict=False)
        _missing = list(getattr(_r, "missing_keys", []))
        _other = [k for k in _missing if not k.startswith("ycb_")]
        if _other or getattr(_r, "unexpected_keys", []):
            raise RuntimeError(f"[载入不符] 除联合色头外的差异键={_other} 多余键={list(getattr(_r,'unexpected_keys',[]))}"
                               f" ⇒ 不是新机制的问题, 停在这里不读数") from _e
        w(f"  [接线自检] 存点里没有 {len(_missing)} 个联合色头键({','.join(_missing)})"
          " ⇒ 用它们的初始值(正交初始化+门初值0⇒G=1)。因此本次读数是**接线证明**, 不是训练后读数:"
          " 未训的联合路以全强度加进 logits, 分数应当**明显掉**; 若与头关时一模一样, 就说明这条路是惰性的")
    p7.load_state_dict(_sd["p7"])
    _a = _sd.get("args") or {}
    w(f"[只评测] 载入 {os.path.basename(args.exam_only)} | epoch={_sd.get('epoch')} | "
      f"存点exam_acc={float(_sd.get('exam_acc', -1))*100:.2f}% | 池={GENERIC_DIR}")
    if not _a:
        w("  [口径未知] 存点里没记 args, 无法校验级别, 读数仅供参考")
    else:
        _bad = [f"{_k}: 存点={_a[_k]} 本次={getattr(args, _k)}" for _k in
                ("pure_learn_ops", "task_cond", "spatial_upgrade", "gate_pair", "transform_v2",
                 "s_seamless", "n_bins", "encoder_width",
                 "color_aux", "gate_span", "p3v_chroma", "p3v_allow_no_skimage", "gen_max",
                 "exam_n_generic", "k_labeled", "k_generic", "batch_size", "epochs", "lr",
                 "ycbcr_head", "ycbcr_aux", "chroma_span", "prompt_cond", "mult_compose", "mult_oracle", "mult_mix",
                 "mult_convex")
                if _k in _a and str(_a[_k]) != str(getattr(args, _k))]
        # [守卫自查] 键列表必须含"我这次打算为测量而改的那个轴"。首版漏了 color_aux, 于是
        #  拿 --color_aux 10 去读一个 color_aux=0 训出的存点时仍打"口径一致" —— 测量有效(不动权重)但声明为假。
        #  例外: decode_probe 故意不进键列表 —— 它只加打印, 不动损失/结构/权重, 加它会让所有旧存点假报警。
        #  对照实测: ycbcr_head 在列表内(它改结构), color_aux 在列表内(它改损失)。
        w("  [口径不符] " + " | ".join(_bad) + " ⇒ 结构对不上会响亮报错或读出错数, 先核对再采信") if _bad \
          else w("  [口径一致] 存点记录的开关与本次 argv 逐项相同")
    exam_eval("(只评测)")
    w("[只评测结束] 未进训练 (刻意不写 [完成] 二字, 免被看门狗误判为该臂已跑完)")
    log.close()
    sys.exit(0)

set_sigma_pool(1)                  # E4 课程从 σ0.1 起步
train_cache = build_train()
exam_eval("(训练前基线)")

best_exam_acc = -1.0
_ckpt_path = os.path.join(SAVE_DIR, "V22_I2I_M2_best.pt")
_inc_args = {}                      # 现存量点的口径, 用来判"这条臂有没有资格夺主线"
if os.path.exists(_ckpt_path):     # [修复] 全局最佳保护: 差于历史最佳的运行不再覆盖checkpoint
    try:
        _inc = torch.load(_ckpt_path, map_location="cpu", weights_only=True)
        best_exam_acc = float(_inc.get("exam_acc", -1.0))
        _inc_args = _inc.get("args") or {}
    except Exception: pass
    w(f"[checkpoint] 历史最佳考试acc={best_exam_acc*100:.2f}% (只被**同口径**的更优运行覆盖)")
    if _inc_args:
        w(f"  [现存量点口径] bs={_inc_args.get('batch_size')} epochs={_inc_args.get('epochs')} "
          f"gen_max={_inc_args.get('gen_max')} lr={_inc_args.get('lr')} "
          f"PL={_inc_args.get('pure_learn_ops')} ycbcr={_inc_args.get('ycbcr_head')} "
          f"| 本次 bs={args.batch_size} epochs={args.epochs} gen_max={args.gen_max} ycbcr={args.ycbcr_head}")
n_nan = 0; n_oom = 0
n_steps_total = 0; n_idle_ep = 0    # [假成功守卫] 一轮都没走通步的 epoch 计数, 见下面"空转"分支
t_start = time.time()
if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()

for ep in range(1, args.epochs + 1):
    if ep > 1 and (ep - 1) % args.regen_every == 0:
        set_sigma_pool(ep)         # E4: 课程档位随 epoch 推进
        train_cache = None;        # [修复] 先放旧缓存再建新, 避免 GPU 上双份共存+碎片
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        train_cache = build_train()
    rgb_all, dgb_all, bin_all, ap_all, ai_all, task_all, sig_all, mf_all, pr_all, pv_all, tc_all = train_cache
    n = rgb_all.shape[0]
    if args.batch_size >= n:
        idx = torch.randint(0, n, (args.batch_size,),
                            generator=torch.Generator().manual_seed(SEED * 13 + ep)).to(DEVICE)
        batches = [idx]
    else:
        perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(SEED * 77 + ep), device="cpu").to(DEVICE)
        batches = [perm[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
    ep_loss = ep_acc = 0.0; steps = 0; consec_err = 0
    for idx in batches:
        if args.v2_probe_every:
            gen.v2_probe = (steps == 0 and (ep - 1) % max(1, args.v2_probe_every) == 0)   # 每N轮的首批打一次(N=1 时即每轮)分支级探针
        if args.s_seamless and args.seam_probe_every:
            gen.seam_probe = (steps == 0 and (ep - 1) % max(1, args.seam_probe_every) == 0)
        try:
            loss, acc, _ = run_batch(rgb_all[idx], dgb_all[idx], bin_all[idx], ap_all[idx], ai_all[idx],
                                     task_all[idx], sig_all[idx], mf_all[idx], pv_all[idx], tgt_cont=tc_all[idx])
        except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
            torch.cuda.empty_cache(); n_oom += 1; opt.zero_grad(); consec_err += 1
            # 60 字截断把最该有的信息切掉了(本轮因此误判过一次"是我的新头吃显存")：
            # 完整 message 里有 "本进程已持有 / 物理还剩 / 上限是多少", 只有这三样能区分"卡真满了"与"自己撞上限"
            _fr2, _tot2 = torch.cuda.mem_get_info()
            w(f"  [E{ep}] 批次异常跳过({type(e).__name__}, 累计{n_oom}): {str(e)[:220]} "
              f"| 本进程持有{torch.cuda.memory_reserved()/2**30:.2f}GB 上限{_eff_cap:.1f}GB 物理空闲{_fr2/2**30:.2f}GB")
            if consec_err > 30:
                w(f"  [E{ep}] 连续30批异常, 疑CUDA上下文损坏, 弃本epoch")
                break
            continue
        consec_err = 0
        if not torch.isfinite(loss):
            n_nan += 1; opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip_norm)
        opt.step()
        ep_loss += loss.item(); ep_acc += acc; steps += 1
        if getattr(gen, 'seam_probe', False) and hasattr(gen, '_seam_stats'):
            st = gen._seam_stats; gen.seam_probe = False
            w(f"          [无缝邻域探针 E{ep}] 注入增益={st['gain']:.4f} 方向选择熵={st['ent']:.3f}(上限{st['ent_max']:.3f}) "
              f"平均最大方向权重={st['wmax']:.3f} seam幅度std={st['seam']:.3f} | 各方向权重={st['w']}")
        if getattr(gen, 'v2_probe', False) and hasattr(gen, '_v2_stats'):
            st = gen._v2_stats; gen.v2_probe = False
            w(f"  [v2探针 E{ep}] 路由熵={st['r_ent']:.3f}(上限{float(np.log(16)):.3f}) 平均最大支路权重={st['r_max']:.3f} "
              f"| copy门g={st['g']:.3f}±{st['g_sd']:.3f} | 合成logits std={st['s_sd']:.2f} copy路std={st['c_sd']:.2f} "
              f"| 用到类={st['used']}/{args.n_bins} | 16支路权重={st['br']}")
            w(f"          [v2分段std] 查表q_in={st['q_sd']:.3f} 邻居混合acc={st['acc_sd']:.3f} "
              f"注意力attn_out={st['attn_sd']:.3f} 身份门控hp={st['hp_sd']:.3f} | 学到的attn增益={st['gain']:.4f}")
    sched.step()
    if DEVICE.type == "cuda" and ep % 25 == 0:   # [回退修复] 运行中动态防护: 桌面挤压时主动归还缓存
        _fr, _ = torch.cuda.mem_get_info()
        _rs = torch.cuda.memory_reserved()
        if _rs > _fr - 0.75 * 2**30:
            torch.cuda.empty_cache()
            w(f"  [显存] 桌面挤压防护: 归还缓存 (持有{_rs/2**30:.1f}GB / 物理空闲{_fr/2**30:.1f}GB)")
    n_steps_total += steps
    if steps == 0:
        n_idle_ep += 1
    if ep % 10 == 0 or ep == 1 or ep == args.epochs:
        eta = (time.time() - t_start) / ep * (args.epochs - ep)
        _vram = torch.cuda.memory_allocated() / 2**20 if DEVICE.type == "cuda" else 0
        w(f"  E{ep}/{args.epochs} loss={ep_loss/max(steps,1):.4f} pixel_acc={ep_acc/max(steps,1)*100:.2f}% | 步数:{steps}/{len(batches)} | GPU常驻{_vram:.0f}MB | {time.time()-t_start:.0f}s ETA{eta:.0f}s | NaN跳步:{n_nan}"
          + ("  ★★空转: 本轮一步都没走通(全批异常跳过), 权重没变过" if steps == 0 else ""))
    if ep % args.exam_every == 0 or ep == args.epochs:
        try:
            acc_e = exam_eval(f"E{ep}", save_png=bool(args.save_png) and ep == args.epochs)
        except torch.OutOfMemoryError:                 # 桌面应用挤显存时跳过该次考试, 不炸训练
            torch.cuda.empty_cache()
            w(f"  [考试E{ep}] OOM跳过(桌面显存挤压)")
            acc_e = -1.0
        if acc_e > best_exam_acc:
            # [夺主线闸] 分数高不等于有权覆盖冠军: 本轮就是把 bs8/12轮/ycbcr_head=1 的 68.21%
            #  判成"更优", 盖掉了 bs16/400轮 的冠军权重(34,233,575B, 65.226%, 全盘再无副本)。
            #  批量、轮数预算、数据池、结构开关都是**分数口径轴**, 任一条不同就只能存成臂名文件, 不碰冠军路径。
            _diff = [f"{_k}: 存量点={_inc_args[_k]} 本次={getattr(args, _k)}" for _k in
                     ("batch_size", "epochs", "gen_max", "lr", "exam_n_generic", "k_labeled", "k_generic",
                      "pure_learn_ops", "task_cond", "spatial_upgrade", "gate_pair", "transform_v2",
                      "s_seamless", "ycbcr_head", "chroma_span", "color_aux", "p3v_chroma", "seed")
                     if _k in _inc_args and str(_inc_args[_k]) != str(getattr(args, _k))]
            if _inc_args and _diff:
                w(f"  [不夺主线] acc={acc_e*100:.2f}% 高于存量点 {best_exam_acc*100:.2f}%, 但口径不同 ⇒ "
                  "这两个数不可比, 冠军权重不动: " + " | ".join(_diff))
                ckpt = os.path.join(SAVE_DIR, f"V22_I2I_M2_arm_bs{args.batch_size}E{ep}.pt")
            else:
                best_exam_acc = acc_e
                ckpt = os.path.join(SAVE_DIR, "V22_I2I_M2_best.pt")
                if os.path.exists(ckpt):   # 覆盖前一律留底(本轮丢的冠军就是缺这一步)
                    _bk = f"{ckpt}.bak_replaced_{time.strftime('%Y%m%d_%H%M%S')}"
                    try:
                        import shutil as _sh; _sh.copy2(ckpt, _bk)
                        w(f"  [留底] 旧最佳已复制到 {os.path.basename(_bk)}")
                    except OSError as _e2:
                        w(f"  [留底失败] {_e2} ⇒ 为安全起见本次不覆盖冠军路径")
                        ckpt = os.path.join(SAVE_DIR, f"V22_I2I_M2_arm_bs{args.batch_size}E{ep}.pt")
            torch.save({"visual_embed": visual_embed.state_dict(), "gen": gen.state_dict(),
                        "p7": p7.state_dict(), "args": vars(args), "epoch": ep,
                        "exam_acc": acc_e, "n_bins": args.n_bins}, ckpt)
            w(f"  [保存] 最佳考试acc={acc_e*100:.2f}% E{ep} → {ckpt}")

_peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0.0
if n_steps_total == 0:
    # [假成功守卫] 本轮就是这么骗过我的: 310 批全 OOM 跳过、一步没走, 末尾照样打出
    #  "[完成] … 最佳考试acc=65.23%"(那是**全局历史最佳**的回显, 不是本臂的读数)。
    #  现在: 一步没走就绝不写 [完成], 退出码 4, 并把真实原因点名。
    w(f"\n[空转失败] {args.epochs} 轮共走通 {n_steps_total} 步(异常跳过 {n_oom} 批) ⇒ 权重从头到尾没变过。")
    w("           这一条 run 不产生任何读数, 不可当'跑完了'记账; 先看上面的 OOM 行里的 持有/上限/物理空闲 三个数。")
    w(f"[日志] {LOG_PATH}")
    log.close()
    sys.exit(4)
if n_idle_ep:
    w(f"  [警告] 有 {n_idle_ep}/{args.epochs} 轮是空转轮(0 步) ⇒ 这些轮的'分数'只是上一轮权重的重复读数")
w(f"\n[完成] 总耗时 {time.time()-t_start:.0f}s | 本轮实际走通 {n_steps_total} 步 | 最佳考试acc={best_exam_acc*100:.2f}% | NaN跳步:{n_nan} | GPU峰值{_peak:.2f}GB")
w(f"[日志] {LOG_PATH}")
log.close()
