"""T5 文本编码器 pread 加载 — 照 122B 绕开 safetensors mmap 段错误"""
import sys, io, os, struct, json, gc, threading
sys.path.insert(0, r'D:\ai')
import torch

DTYPE_MAP={'BF16':torch.bfloat16,'F32':torch.float32,'F16':torch.float16,'I64':torch.int64,'BOOL':torch.bool,'I32':torch.int32}

_tl=threading.local()
def _fh(path):
    h=getattr(_tl,'h',None)
    if h is None: h={}; _tl.h=h
    if path not in h: h[path]=open(path,'rb')
    return h[path]

def load_t5_shard_weights(shard_path):
    """pread 读取一个 shard 的全部权重"""
    out={}
    with open(shard_path,'rb') as f:
        hs=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(hs))
        base=8+hs
    for k,meta in header.items():
        if k=='__metadata__': continue
        o0,o1=meta['data_offsets']
        f=_fh(shard_path)
        f.seek(base+o0)
        buf=bytearray(f.read(o1-o0))
        out[k]=torch.frombuffer(buf,dtype=DTYPE_MAP[meta['dtype']]).reshape(meta['shape'])
    return out

def load_t5_pread(model_dir, dtype=torch.bfloat16):
    """用 pread 加载 T5EncoderModel"""
    import glob, os
    from transformers import T5Config, T5EncoderModel
    cfg=T5Config.from_pretrained(model_dir,local_files_only=True)
    model=T5EncoderModel(cfg)
    # 收集所有 shard 权重
    all_w={}
    shards=sorted(glob.glob(os.path.join(model_dir,'model-*.safetensors')))
    for sh in shards:
        print(f'[t5-pread] {os.path.basename(sh)}',flush=True)
        w=load_t5_shard_weights(sh)
        for k,v in w.items():
            all_w[k]=v.to(dtype)
        del w; gc.collect()
    # 填充模型
    model.load_state_dict(all_w, strict=False)
    print(f'[t5-pread] loaded {len(all_w)} keys',flush=True)
    del all_w; gc.collect()
    return model

if __name__=='__main__':
    m=load_t5_pread(r'D:\ai\models\LTX-Video\text_encoder')
    print('T5 loaded', m.config.hidden_size, flush=True)
