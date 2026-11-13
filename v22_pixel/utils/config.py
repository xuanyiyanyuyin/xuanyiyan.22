# utils/config.py — V18 v5.0 非对称映射 (编码2048D → 解码128D)
import torch, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WORD_LIST_PATH = os.path.join(DATA_DIR, "word_list_v2.txt")
SAVE_DIR = os.path.join(BASE_DIR, "P1_char_word", "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === P1 内部高维编码 (2048D + 512头) ===
P1_CONTENT_DIM = 2040
P1_CHAR_DIM = 8 + P1_CONTENT_DIM       # 2048
P1_HEADS = 512
P1_HEAD_DIM = P1_CHAR_DIM // P1_HEADS   # 4

# === 下游统一维度 (128D + 64头) ===
POS_DIM = 8
CONTENT_DIM = 120
CHAR_DIM = POS_DIM + CONTENT_DIM        # 128
WORD_DIM = CHAR_DIM                      # 128
SENT_DIM = CHAR_DIM * 2                 # 256

# === 下游交叉注意力 ===
ATTN_HEADS = 64
ATTN_DIM = 256
ATTN_HEAD_DIM = ATTN_DIM // ATTN_HEADS  # 4
ATTN_DROPOUT = 0.1

# === 调制 ===
P1_MOD_DIM = P1_CHAR_DIM               # 2048
MOD_DIM = WORD_DIM                      # 128

# === 解码器隐层 ===
HIDDEN_DIM = 256

# === 训练 ===
LEARNING_RATE = 0.005
BATCH_SIZE = 200
DEFAULT_EPOCHS = 300
DISPLAY_INTERVAL = 20
PEARSON_EPSILON = 1e-8
WEIGHT_DECAY = 1e-5
DISPLAY_MODE = "average"
PATIENCE = 25
