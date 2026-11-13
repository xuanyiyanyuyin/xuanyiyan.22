import os, sys, time, torch
sys.path.insert(0, r"D:\ai\py_h3")
R = r"D:\ai\models\MiniMax-H3\Ref2VA"
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
t0=time.time()
v = AutoencoderKLMiniMaxH3.from_pretrained(os.path.join(R,"video_vae"), dtype=torch.bfloat16)
print("[V] video_vae OK %.1fs  %s  参数量=%.1fM" % (time.time()-t0, type(v).__name__, sum(p.numel() for p in v.parameters())/1e6), flush=True)
t0=time.time()
a = AutoencoderKLMiniMaxH3Audio.from_pretrained(os.path.join(R,"audio_vae"), dtype=torch.bfloat16)
print("[V] audio_vae OK %.1fs  %s  参数量=%.1fM" % (time.time()-t0, type(a).__name__, sum(p.numel() for p in a.parameters())/1e6), flush=True)
import gc; gc.collect(); os._exit(0)