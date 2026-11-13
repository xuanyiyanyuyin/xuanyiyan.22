import torch, time, sys
sys.path.insert(0, r'D:\ai')
from diffusers import Flux2Pipeline
import flux2_3tier as f2
DEV='cuda'
pipe = Flux2Pipeline.from_pretrained(f2.MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
    low_cpu_mem_usage=True, text_encoder=None)
blk = pipe.transformer.single_transformer_blocks[0].to(DEV).eval()
DIM = blk.attn.to_qkv_mlp_proj.weight.shape[1]
print(f'dim={DIM}', flush=True)
pos_embed = pipe.transformer.pos_embed
# time_guidance_embed / modulation 也常驻 GPU (默认在CPU)
pipe.transformer.time_guidance_embed = pipe.transformer.time_guidance_embed.to(DEV)
pipe.transformer.single_stream_modulation = pipe.transformer.single_stream_modulation.to(DEV)

def make_inputs(S):
    ids = torch.zeros(S, 4, dtype=torch.int64, device=DEV)
    idx = torch.arange(S, device=DEV)
    ids[:,1] = idx % 160
    ids[:,2] = (idx // 160) % 160
    cos, sin = pos_embed(ids)
    hidden = torch.randn(1, S, DIM, device=DEV, dtype=torch.bfloat16)
    temb = pipe.transformer.time_guidance_embed(
        torch.tensor([500.0], dtype=torch.bfloat16, device=DEV),
        torch.tensor([4.0], dtype=torch.bfloat16, device=DEV))
    mod = pipe.transformer.single_stream_modulation(temb)
    return hidden, (cos, sin), mod

for S in (2560, 51712):
    hidden, rot, mod = make_inputs(S)
    with torch.no_grad():
        _ = blk(hidden_states=hidden, encoder_hidden_states=None, temb_mod=mod, image_rotary_emb=rot)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(2):
            out = blk(hidden_states=hidden, encoder_hidden_states=None, temb_mod=mod, image_rotary_emb=rot)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 2
        print(f'S={S}: 单块前向 avg {dt:.2f}s', flush=True)
        del hidden, rot, mod, out
        torch.cuda.empty_cache()
print('BLOCK_TIME_DONE', flush=True)
