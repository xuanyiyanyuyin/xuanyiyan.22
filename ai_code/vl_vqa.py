"""
Qwen3-VL-30B 视觉问答 — 官方模型, 看图说话
架构: 27层ViT视觉编码器 + 48层MoE文本(3B激活)
显存: device_map=auto, 视觉上GPU, 文本层溢CPU/RAM
用法: D:\ai\python\python.exe vl_vqa.py <图片路径> [问题]
      (默认问题: 请描述这张图片的内容)
"""
import torch, sys, time, os
sys.path.insert(0, r'D:\ai')
os.environ['HF_HUB_OFFLINE']='1'

MODEL=r'D:\ai\models\Qwen3-VL-30B-A3B-Instruct'

_model=None
_processor=None

def get_model():
    global _model, _processor
    if _model is not None: return _model, _processor
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
    t0=time.perf_counter()
    _processor=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    # 三层存储: GPU 6GB(视觉+非专家+激活) → CPU 16GB(专家缓存) → Disk(offload)
    # CPU 上限从 25GiB 降到 16GiB：本机 32GB RAM，NapCat 等常驻 ~9GB，25GB 会 OOM 崩进程
    _model=Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
        device_map='auto', max_memory={0:'6GiB','cpu':'16GiB'},
        trust_remote_code=True, local_files_only=True,
        offload_folder=r'D:\ai\offload',
        offload_state_dict=True)
    print(f'[vl] model ready {time.perf_counter()-t0:.0f}s', flush=True)
    return _model, _processor

def ask_image(image_path, question='请用中文详细描述这张图片的内容。'):
    """看图回答问题, 返回文本"""
    from PIL import Image
    model, processor = get_model()
    img=Image.open(image_path).convert('RGB')
    messages=[{'role':'user','content':[
        {'type':'image','image':img},
        {'type':'text','text':question}
    ]}]
    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs=processor(text=[text], images=[img], return_tensors='pt')
    device=model.device if hasattr(model,'device') else next(model.parameters()).device
    inputs={k:(v.to(device) if hasattr(v,'to') else v) for k,v in inputs.items()}
    t0=time.perf_counter()
    with torch.inference_mode():
        gen=model.generate(**inputs, max_new_tokens=256, do_sample=False)
    out=processor.decode(gen[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'[vl] answered in {time.perf_counter()-t0:.0f}s', flush=True)
    return out

def free():
    global _model, _processor
    del _model; del _processor
    _model=None; _processor=None
    torch.cuda.empty_cache()
    print('[vl] released', flush=True)

if __name__=='__main__':
    path=sys.argv[1] if len(sys.argv)>1 else r'D:\ai\output\images\chat_1786343812.png'
    q=sys.argv[2] if len(sys.argv)>2 else '请用中文详细描述这张图片的内容。'
    if not os.path.exists(path):
        print(f'图片不存在: {path}'); sys.exit(1)
    ans=ask_image(path, q)
    print(f'[answer] {ans}', flush=True)
    free()
