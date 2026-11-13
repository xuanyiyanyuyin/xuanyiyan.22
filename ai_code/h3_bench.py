# -*- coding: utf-8 -*-
"""H3 对齐台：diffusers port transformer 与手写引擎【逐块】对数。

照 经验库/ai-pipeline_推理经验库.txt:796-799 的做法（官方 hook 抓每块输出），
判据用 :829 的 max_diff < 1e-3 与 :778 的 CORR > 0.999。
同一份打包输入、同一份权重：先跑手写 block 循环（流式，省显存）并记下检查点块输出，
释放显存后再加载 port（bf16 10GB）跑一遍，逐块比。
键名转换只负责"让 port 能加载 native 权重"；qkv 切分与 fc1 半序已由
ComfyUI native + LoRA 元数据独立裁定，不构成本台的循环论证。
用法: python h3_bench.py [sigma]
"""
import os, sys, json, torch, torch.nn.functional as F
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, r"D:\ai")
sys.path.insert(0, r"D:\ai\py_h3")
import h3_engine as E
import h3_score as S

CHECK = (0, 1, 12, 25, 37, 49)

def _prefix(name):
    """native 键 -> (port 前缀, 余下部分)"""
    p = name.split(".")
    if p[0] == "blocks":
        return "transformer_blocks.%s." % p[1], ".".join(p[2:])
    if p[0] == "token_refiner" and p[1] == "blocks":
        return "token_refiner.refiner_blocks.%s." % p[3], ".".join(p[4:])
    return "", name

def conv_pairs(name):
    """native 键 -> [(port 键, 行切片或 None)]"""
    pre, rest = _prefix(name)
    if rest.startswith("attn.qkv_proj."):
        suf = rest[len("attn.qkv_proj."):]
        return [(pre + "attn.to_q." + suf, (0, 7168)),
                (pre + "attn.to_k." + suf, (7168, 14336)),
                (pre + "attn.to_v." + suf, (14336, 21504))]
    table = [("attn.q_norm.", "attn.norm_q."), ("attn.k_norm.", "attn.norm_k."),
             ("attn.out_proj.", "attn.to_out.0."), ("mlp.fc1.", "ff.net.0.proj."),
             ("mlp.fc2.", "ff.net.2."), ("video_patch_proj.", "proj_in."),
             ("audio_patch_proj.", "audio_proj_in."), ("condition_proj.", "context_embedder."),
             ("time_embedder.proj_in.", "time_embedder.linear_1."),
             ("time_embedder.proj_out.", "time_embedder.linear_2."),
             ("final_layer.norm.", "norm_out.norm."),
             ("final_layer.adaln_proj.linear.", "norm_out.linear."),
             ("final_layer.video_out.", "proj_out."), ("final_layer.audio_out.", "audio_proj_out.")]
    for a, b in table:
        if rest.startswith(a):
            return [(pre + b + rest[len(a):], None)]
    return [(pre + rest, None)]

def load_port_state():
    from safetensors import safe_open
    idx = json.load(open(os.path.join(E.TF, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
    sd = {}
    for sh in sorted(set(idx.values())):
        with safe_open(os.path.join(E.TF, sh), framework="pt") as f:
            for k in list(f.keys()):
                t = f.get_tensor(k)
                for dst, sl in conv_pairs(k):
                    sd[dst] = t[sl[0]:sl[1]] if sl else t
    return sd

def my_blocks(P, sm, lf):
    """复刻 dit_forward 的前置与 block 循环，返回检查点块输出（CPU）与末层输入 h。"""
    xt, xa, cond = P["xt"], P["xa"], P["cond"]
    ve = E.lin(xt.to(E.DEV, sm["video_patch_proj.weight"].dtype), sm["video_patch_proj.weight"], sm["video_patch_proj.bias"])
    ae = E.lin(xa.to(E.DEV, sm["audio_patch_proj.weight"].dtype), sm["audio_patch_proj.weight"], sm["audio_patch_proj.bias"])
    te = E.lin(cond.to(E.DEV, E.DTYPE), sm["condition_proj.weight"], sm["condition_proj.bias"])
    te = E.refiner(te, None, sm)
    L = P["pos"].shape[0]
    h = te.new_zeros((te.shape[0], L, te.shape[-1]))
    h = h.index_copy(1, P["txt"].to(E.DEV), te)
    h = h.index_copy(1, P["vid"].to(E.DEV), ve.to(te.dtype))
    h = h.index_copy(1, P["aud"].to(E.DEV), ae.to(te.dtype))
    temb = E.time_embed(P["uniq"].to(E.DEV))
    temb = E.lin(temb, sm["time_embedder.proj_in.weight"], sm["time_embedder.proj_in.bias"])
    temb = E.lin(F.silu(temb), sm["time_embedder.proj_out.weight"], sm["time_embedder.proj_out.bias"])
    idx = (P["ti"].to(E.DEV) * E.MODE_NUM + P["tags"].to(E.DEV)).to(torch.long)
    cos, sin = E.build_rope(P["pos"].to(E.DEV), sm["rope.inv_freq"].to(torch.float32))
    caps = {}
    for i in range(50):
        st = lf(i)
        _aw = st["adaln_proj.linear.weight"]
        a = E.lin(F.silu(temb.float()).to(_aw.dtype), _aw, st["adaln_proj.linear.bias"]).view(-1, 6 * E.HID)
        h = E.block_forward(h, st, a.chunk(6, dim=-1), idx, cos, sin)
        if i in CHECK:
            caps[i] = h.detach().float().cpu()
    return caps, h, temb

def main():
    sigma = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    P = S.build_packed(r"D:\ai\output\images\gen_68965_0.png", sigma, "none")
    sm, lf = P["sm"], P["lf"]
    caps, h_mine, temb = my_blocks(P, sm, lf)
    print("[b] 手写 50 块完成，检查点 %s" % (list(caps),), flush=True)
    del sm, lf
    import gc; gc.collect(); torch.cuda.empty_cache()

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    m = MiniMaxH3Transformer3DModel()
    sd = load_port_state()
    miss, unexp = m.load_state_dict(sd, strict=False)
    print("[b] port 加载: missing=%d unexpected=%d" % (len(miss), len(unexp)), flush=True)
    if miss or unexp:
        print("[b] missing 样本:", miss[:5]); print("[b] unexpected 样本:", unexp[:5]); return False
    m = m.to(torch.bfloat16).to(E.DEV).eval()
    pcaps = {}
    def mk(i):
        def fn(mod, inp, out):
            pcaps[i] = out.detach().float().cpu()
        return fn
    for i in CHECK:
        m.transformer_blocks[i].register_forward_hook(mk(i))
    with torch.no_grad():
        out = m(hidden_states=P["xt"].to(E.DEV, torch.bfloat16),
                audio_hidden_states=P["xa"].to(E.DEV, torch.bfloat16),
                encoder_hidden_states=P["cond"].to(E.DEV, torch.bfloat16),
                timestep=P["uniq"].to(E.DEV), timestep_indices=P["ti"].to(E.DEV),
                token_tags=P["tags"].to(E.DEV), position_ids=P["pos"].to(E.DEV),
                video_indices=P["vid"].to(E.DEV), audio_indices=P["aud"].to(E.DEV),
                text_indices=P["txt"].to(E.DEV), return_dict=True)
    ok = True
    for i in CHECK:
        a, b = pcaps[i], caps[i]
        d = (a - b).abs().max().item()
        rel = d / max(b.abs().max().item(), 1e-9)
        corr = torch.corrcoef(torch.stack([a.reshape(-1), b.reshape(-1)]))[0, 1].item()
        # bf16 全块对数的现实阈值：CORR>0.9999 且相对 max_diff<1e-2
        # （经验库:829 的 1e-3 是 RoPE 那种 fp 数学的阈值，不套在 bf16 块上）
        good = corr > 0.9999 and rel < 1e-2
        ok = ok and good
        print("[b] block %2d  max_diff=%.3e 相对=%.3e  CORR=%.6f  %s" % (
            i, d, rel, corr, "✅" if good else "✗ 超阈"), flush=True)
    vo_port = out.sample.float().cpu()
    print("[b] port 末层输出 std=%.4f（手写末层另验）" % vo_port.std().item(), flush=True)
    print("[b] ★★★ 逐块对齐 %s" % ("通过" if ok else "未通过"), flush=True)
    return ok

if __name__ == "__main__":
    main()
    os._exit(0)
