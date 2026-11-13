"""
GPU=计算器，不是存储器。逐tensor加载→计算→释放。
每个时刻GPU只放1-2个权重(几MB)，算完立刻丢回RAM。
"""
import torch, torch.nn.functional as F, time, numpy as np, sys, json, os, struct, threading, queue

MODEL_DIR=r'D:\ai\models\qwen3.5-397b-bf16'; DEV="cuda"
sys.path.insert(0,r'D:\ai')
from expert_reader import SafeExpertReader
from non_expert_streamer import NonExpertStreamer
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule,l2norm

tk=AutoTokenizer.from_pretrained(MODEL_DIR,trust_remote_code=True)
cf=AutoConfig.from_pretrained(MODEL_DIR,trust_remote_code=True).text_config
st=NonExpertStreamer(MODEL_DIR); rd=SafeExpertReader(MODEL_DIR)
H=cf.hidden_size; V=cf.vocab_size; LV=cf.layer_types
NV=64; NK=16; HK=128; HV=128; EP=cf.rms_norm_eps

# ── Embed ──
with open(MODEL_DIR+'/model.safetensors.index.json') as f: wm=json.load(f)['weight_map']
ek=[k for k in wm if 'language_model' in k and 'embed_tokens' in k][0]
with open(MODEL_DIR+'/'+wm[ek],'rb') as f: n=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(n))
m=hdr[ek]; s,e=m['data_offsets']; fd=os.open(MODEL_DIR+'/'+wm[ek],os.O_RDONLY|os.O_BINARY)
os.lseek(fd,n+8+s,os.SEEK_SET); emb=torch.from_numpy(np.frombuffer(os.read(fd,e-s),dtype=np.int16).copy()).view(torch.bfloat16).reshape(V,H).float().to(DEV); os.close(fd)

# ── 单tensor加载: 从numpy到GPU, 用完即丢 ──
def load1(arr):
    """加载单个权重numpy→GPU, 调用方用完立刻del"""
    if arr.dtype==np.float32: return torch.from_numpy(arr.copy()).to(DEV)
    return torch.from_numpy(arr.view(np.uint16).astype(np.int16).copy()).view(torch.bfloat16).to(DEV)

def getw(ne,pf,suffix):
    """从numpy dict取一个权重→GPU, 返回(GPU_tensor, key)"""
    k=pf+suffix; v=ne.get(k)
    return load1(v) if v is not None else None

# ── KV Cache ──
class KVC:
    def __init__(self): self.kv={}; self.cv={}; self.rc={}
    def gkv(self,l): return self.kv.get(l)
    def gcv(self,l): return self.cv.get(l)
    def grc(self,l): return self.rc.get(l)
    def ukv(self,l,k,v):
        if l in self.kv: self.kv[l]=(torch.cat([self.kv[l][0],k],2),torch.cat([self.kv[l][1],v],2))
        else: self.kv[l]=(k,v)
    def ucv(self,l,s): self.cv[l]=s
    def urc(self,l,s): self.rc[l]=s

# ── Prefetch thread ──
pre_q=queue.Queue(maxsize=2)
def prefetch():
    for lid in range(60): pre_q.put((lid,st.read_layer(lid)))

# ── Generate ──
def gen(prompt):
    ids=torch.tensor([tk.encode(prompt)],device=DEV)
    h=F.embedding(ids,emb); cache=KVC(); g=ids[0].tolist()

    for step in range(3):
        pre=(step==0); t0=time.perf_counter()
        thr=threading.Thread(target=prefetch,daemon=True); thr.start()

        for lid in range(60):
            lid2,ne=pre_q.get(); assert lid2==lid
            pf=f'model.language_model.layers.{lid}.'
            bt=LV[lid]
            B,S,_=h.shape

            with torch.no_grad():
                # 1) RMSNorm
                inw=getw(ne,pf,'input_layernorm.weight')
                hn=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
                if inw is not None: hn=hn*inw.float(); del inw

                # 2) Attention — load/calc/free per tensor
                if bt=="full_attention":
                    q=getw(ne,pf,'self_attn.q_proj.weight')
                    k=getw(ne,pf,'self_attn.k_proj.weight')
                    v=getw(ne,pf,'self_attn.v_proj.weight')
                    o=getw(ne,pf,'self_attn.o_proj.weight')
                    if q is not None:
                        Q=hn.float()@q.float().T; del q
                        K=hn.float()@k.float().T; del k
                        V=hn.float()@v.float().T; del v
                        qa=Q[:,:,:8192].view(B,S,32,256).transpose(1,2)
                        qg=Q[:,:,8192:].view(B,S,32,256).transpose(1,2); del Q
                        K=K.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
                        V=V.view(B,S,2,256).transpose(1,2).repeat_interleave(16,1)
                        if not pre:
                            kc,vc=cache.gkv(lid)
                            K=torch.cat([kc,K],2) if kc is not None else K
                            V=torch.cat([vc,V],2) if vc is not None else V
                        cache.ukv(lid,K,V)
                        att=F.scaled_dot_product_attention(qa,K,V,is_causal=pre)
                        a_out=((torch.sigmoid(qg)*att).transpose(1,2).contiguous().view(B,S,8192)@o.float().T).to(hn.dtype)
                        del qa,qg,K,V,att,o
                    else: a_out=hn*0.99
                else:
                    qkv=getw(ne,pf,'linear_attn.in_proj_qkv.weight')
                    z_w=getw(ne,pf,'linear_attn.in_proj_z.weight')
                    a_w=getw(ne,pf,'linear_attn.in_proj_a.weight')
                    b_w=getw(ne,pf,'linear_attn.in_proj_b.weight')
                    o_w=getw(ne,pf,'linear_attn.out_proj.weight')
                    cv=getw(ne,pf,'linear_attn.conv1d.weight')
                    dt_w=getw(ne,pf,'linear_attn.dt_bias')
                    al_w=getw(ne,pf,'linear_attn.A_log')
                    nr=getw(ne,pf,'linear_attn.norm.weight')
                    if qkv is not None:
                        KD=NK*HK; VD=NV*HV
                        m=hn.float()@qkv.float().T; del qkv
                        m=m.transpose(1,2)
                        if cv is not None:
                            cw=cv.float().squeeze(1) if cv.dim()==4 else cv.float(); del cv
                            if not pre: cs=cache.gcv(lid); m=torch.cat([cs,m],-1) if cs is not None else m
                            m=F.silu(F.conv1d(m,cw,padding=3,groups=12288))[:,:,:S]
                            cache.ucv(lid,m[:,:,-3:].clone())
                        m=m.transpose(1,2)
                        Q=m[:,:,:KD].reshape(B,S,NK,HK); K=m[:,:,KD:2*KD].reshape(B,S,NK,HK)
                        V=m[:,:,2*KD:].reshape(B,S,NV,HV); del m
                        beta=(hn.float()@b_w.float().T).sigmoid() if b_w is not None else torch.ones(B,S,NK,device=DEV)
                        av=(hn.float()@a_w.float().T) if a_w is not None else torch.zeros(B,S,NV,device=DEV)
                        g=-al_w.float().exp()*F.softplus(av+(dt_w.float() if dt_w is not None else 0))
                        del a_w,b_w,al_w,dt_w,av
                        R=NV//NK
                        Q=Q.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK)
                        K=K.unsqueeze(2).expand(-1,-1,R,-1,-1).reshape(B,S,NV,HK)
                        qn=l2norm(Q,-1,1e-6); kn=l2norm(K,-1,1e-6)
                        init=None if pre else cache.grc(lid)
                        ca,ls=torch_chunk_gated_delta_rule(qn,kn,V,g,beta,64,init,True,True)
                        cache.urc(lid,ls); del qn,kn,V,g,beta,Q,K,ls
                        at=ca.float().transpose(1,2).contiguous().view(B,S,VD); del ca
                        if nr is not None:
                            ni=at.reshape(B,S,NV,HV)
                            at=(ni*torch.rsqrt(ni.pow(2).mean(-1,keepdim=True)+EP)*nr.float().view(1,1,1,-1)).reshape(B,S,VD)
                            del ni,nr
                        zz=(hn.float()@z_w.float().T) if z_w is not None else at; at=at*F.silu(zz); del zz,z_w
                        a_out=((at@o_w.float().T) if o_w is not None else at[:,:,:H]).to(hn.dtype)
                        del at,o_w
                    else: a_out=hn*0.99
                h=(h+a_out).to(h.dtype); del a_out,hn

                # 3) RMSNorm + MoE
                pn=getw(ne,pf,'post_attention_layernorm.weight')
                sgw=getw(ne,pf,'mlp.shared_expert_gate.weight')
                sg_=getw(ne,pf,'mlp.shared_expert.gate_proj.weight')
                su=getw(ne,pf,'mlp.shared_expert.up_proj.weight')
                sd=getw(ne,pf,'mlp.shared_expert.down_proj.weight')
                rw=getw(ne,pf,'mlp.gate.weight')

                hn2=h*torch.rsqrt(h.float().pow(2).mean(-1,keepdim=True)+EP)
                if pn is not None: hn2=hn2*pn.float(); del pn

                flat=hn2.to(torch.bfloat16).reshape(-1,H)
                logits=flat.float()@rw.float().T; probs=torch.softmax(logits,-1); del logits
                _,ti=torch.topk(probs,10,-1); uids=ti.unique().tolist()[:10]

                moe_out=torch.zeros_like(flat,dtype=torch.float32)
                for eid in uids:
                    gn,dn=rd.read_expert(lid,eid)
                    g2=torch.from_numpy(gn.astype(np.int16).copy()).view(torch.bfloat16).float().to(DEV)
                    d2=torch.from_numpy(dn.astype(np.int16).copy()).view(torch.bfloat16).float().to(DEV)
                    mask=(ti==eid).any(-1)
                    if not mask.any(): del g2,d2; continue
                    t=flat[mask].float()
                    o=(F.silu(t@g2[:1024].T)*(t@g2[1024:].T))@d2.T
                    tw=probs[mask]; eh=(ti[mask]==eid)
                    moe_out[mask]+=o*(tw*eh).sum(-1,keepdim=True); del g2,d2,o,t,tw,eh

                gv=torch.sigmoid(flat.float()@sgw.float().T); del sgw,flat
                so=(F.silu(hn2.float().reshape(-1,H)@sg_.float().T)*(hn2.float().reshape(-1,H)@su.float().T))@sd.float().T
                del sg_,su,sd,hn2
                moe_out=moe_out+gv*so; del gv,so,ti,probs
                h=(h+moe_out.to(h.dtype).reshape_as(h)); del moe_out,rw

            if lid%15==0: torch.cuda.empty_cache()

        thr.join()
        logits=h[:,-1:,:].float()@emb.float().T/0.8
        if torch.isnan(logits).any(): logits=torch.nan_to_num(logits,nan=-1e9)
        v,_=torch.topk(logits,40,-1); logits[logits<v[:,:,-1:]]=-float('inf')
        pr=torch.softmax(logits,-1)
        if torch.isnan(pr).any(): pr=torch.ones_like(pr)/V
        nt=torch.multinomial(pr.squeeze(),1).item(); g.append(nt)
        print(f"  [{step}] '{tk.decode([nt])}' ({time.perf_counter()-t0:.0f}s)",flush=True)
        if nt==tk.eos_token_id: break
        if step<2: h=F.embedding(torch.tensor([[nt]],device=DEV),emb).to(h.dtype)

    return tk.decode(g)

for p in ["你好"]:
    print(f"\n{p!r}"); r=gen(p); print(f"  -> {r!r}")

st.close(); rd.close()
print(f"\nVRAM: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
