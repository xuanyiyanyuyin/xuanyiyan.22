import json, os
TF = r"D:\ai\models\MiniMax-H3\Ref2VA\transformer"
wm = json.load(open(os.path.join(TF,"model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
oth = sorted(n for n in wm if not n.startswith("blocks."))
print("=== ckpt 非层权重（%d 个）===" % len(oth))
for n in oth: print("  " + n)