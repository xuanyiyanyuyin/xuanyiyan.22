"""
音频双模型 — 生成基础干声 (精细调音交给外部软件)
1. Qwen3-TTS-12Hz-1.7B-Base: 文字→语音 (voice clone, 合成参考音频)
2. Qwen3-ASR-1.7B: 语音→文字 (拿TTS输出做输入, 闭环验证)
音质: 重采样到 48kHz 专业级 (接近CD/DVD音质)
工作流: AI 出毛坯干声 → Audacity/Praat/RVC 等免费工具精修 (EQ/变调/共振峰)
用法: D:\\ai\\python\\python.exe gen_audio.py [文本] [音色描述(暂未启用)]
"""
import torch, sys, time, os
sys.path.insert(0, r'D:\ai')
TTS_DIR=r'D:\ai\models\Qwen3-TTS-12Hz-1.7B-Base'
ASR_DIR=r'D:\ai\models\Qwen3-ASR-1.7B'
OUT=r'D:\ai\output\audio'
OUT_HI=r'D:\ai\output\audio\hiq'
os.makedirs(OUT, exist_ok=True); os.makedirs(OUT_HI, exist_ok=True)

HI_SR = 48000  # 专业级采样率

def resample_to_hi(wav, sr_in, sr_out=HI_SR):
    """线性插值重采样到 48kHz"""
    import numpy as np
    if sr_in == sr_out:
        return wav
    n_out = int(len(wav) * sr_out / sr_in)
    x_old = np.linspace(0, 1, len(wav), endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)

def main():
    text=sys.argv[1] if len(sys.argv)>1 else '你好, 我是本地部署的千问语音合成模型, 现在给你读一段话。'
    lang='Chinese'

    print('=== TTS: 文字 -> 语音 (声音克隆, 合成参考) ===',flush=True)
    from qwen_tts import Qwen3TTSModel
    import soundfile as sf
    import numpy as np
    t0=time.perf_counter()
    tts=Qwen3TTSModel.from_pretrained(TTS_DIR,device_map='cuda',dtype=torch.bfloat16,attn_implementation='sdpa')
    print(f'[tts] model loaded {time.perf_counter()-t0:.0f}s',flush=True)
    # 合成参考音频 (48kHz)
    sr_ref=48000;t_=np.arange(0,4,1/sr_ref)
    ref=np.sin(2*np.pi*220*t_)+0.5*np.sin(2*np.pi*440*t_)+0.25*np.sin(2*np.pi*660*t_)
    ref*=0.3*(1+0.3*np.sin(2*np.pi*5*t_))
    ref_fn=f'{OUT}/ref_test.wav';sf.write(ref_fn,(ref/ref.max()*0.8).astype(np.float32),sr_ref)
    t1=time.perf_counter()
    wavs,sr=tts.generate_voice_clone(text=text,language=lang,ref_audio=ref_fn,ref_text='啦啦啦啦啦, 测试参考音频。')
    # 重采样到 48kHz 专业级
    wav_hi=resample_to_hi(wavs[0], sr)
    fn=f'{OUT_HI}/tts_hiq_{int(time.time())}.wav'
    sf.write(fn, wav_hi, HI_SR)
    print(f'[tts] done {time.perf_counter()-t1:.1f}s -> {fn}',flush=True)
    print(f'[tts] 音质: {sr}Hz -> {HI_SR}Hz 专业级 ({len(wav_hi)/HI_SR:.1f}s)',flush=True)
    # 保留原始版本(ASR用)
    fn_orig=f'{OUT}/tts_{int(time.time())}.wav'
    sf.write(fn_orig, wavs[0], sr)
    del tts;torch.cuda.empty_cache()

    print('=== ASR: 语音 -> 文字 ===',flush=True)
    from qwen_asr import Qwen3ASRModel
    t0=time.perf_counter()
    asr=Qwen3ASRModel.from_pretrained(ASR_DIR,device_map='cuda',torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    print(f'[asr] model loaded {time.perf_counter()-t0:.0f}s',flush=True)
    res=asr.transcribe(audio=fn_orig,language='Chinese')
    out=res[0].text
    print(f'[asr] 识别结果: {out}',flush=True)
    print(f'[check] 原文: {text}',flush=True)
    print(f'[out] 48kHz高音质: {fn}',flush=True)

if __name__=='__main__':
    main()
