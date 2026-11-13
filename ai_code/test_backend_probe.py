import torch
import sys
sys.path.insert(0, r'D:\ai')
import flux2_3tier as f2
from diffusers import Flux2Pipeline
MODEL = f2.MODEL
pipe = Flux2Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True,
    low_cpu_mem_usage=True, text_encoder=None)
from diffusers.models.attention_dispatch import _AttentionBackendRegistry
blk = pipe.transformer.single_transformer_blocks[0]
proc = blk.attn.processor
print('processor._attention_backend =', repr(proc._attention_backend), flush=True)
print('active_backend =', repr(_AttentionBackendRegistry.get_active_backend()[0]), flush=True)
print('db single proc:', repr(pipe.transformer.transformer_blocks[0].attn.processor._attention_backend), flush=True)
print('PROBE_DONE', flush=True)
