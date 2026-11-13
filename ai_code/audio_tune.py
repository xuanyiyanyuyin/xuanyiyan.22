"""
音频调音 pipeline — 对 AI 干声做确定性后处理 (纯数值参数)
支持: 变调(pitch_shift) / 变速(time_stretch) / EQ(lowpass/highpass) / 增益 / 峰值归一化 / 混响(reverb)
依赖: librosa + scipy + soundfile + numpy (已在 D:\\ai\\python 环境)

用法: D:\\ai\\python\\python.exe audio_tune.py <输入音频> <参数JSON> [输出路径]
参数JSON (全部可选, 未写则跳过该项):
  {
    "pitch_shift": 2,      # 变调 半音 (-12~+12, 保留共振峰)
    "formant_shift": 1.15, # 共振峰移位 (1.0=不变, >1偏年轻/亮, <1偏成熟/暗; Praat)
    "time_stretch": 1.0,   # 变速 (0.5=慢一半, 1.5=快一半)
    "gain_db": 3,          # 音量增益 dB (-20~+20)
    "lowpass": 8000,       # 低通截止 Hz (0=关, 如 8000 去齿音)
    "highpass": 80,        # 高通截止 Hz (0=关, 如 80 去低频嗡)
    "normalize": true,     # 峰值归一化到 -1dB
    "reverb": 0.15,        # 混响量 0~1 (0=关, 卷积混响)
    "target_sr": 48000     # 输出采样率 (可选, 默认保持原样)
  }

输出: 默认 <输入>_tuned.wav, 打印最终路径 (供前端/批处理抓取)
"""
import sys, os, json, time
import numpy as np

try:
    import soundfile as sf
    import librosa
    from scipy.signal import butter, lfilter
except ImportError as e:
    print(f'[error] 缺依赖: {e}', flush=True)
    raise


def _butter_lowpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal = cutoff / nyq
    b, a = butter(order, normal, btype='low')
    return b, a


def _butter_highpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal = cutoff / nyq
    b, a = butter(order, normal, btype='high')
    return b, a


def _apply_reverb(wav, sr, amount):
    """简单卷积混响: 指数衰减噪声冲激响应"""
    if amount <= 0:
        return wav
    dur = 0.6  # 混响时长秒
    n = int(sr * dur)
    rng = np.random.default_rng(42)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 8, n))
    ir[0] = 1.0  # 直达声
    ir = ir / np.abs(ir).sum()
    wet = np.convolve(wav, ir, mode='full')[:len(wav)]
    return (1 - amount) * wav + amount * wet


def _formant_shift(wav, sr, ratio):
    """共振峰移位 (Praat Change gender, 保持音高和时长)"""
    import parselmouth
    snd = parselmouth.Sound(np.asarray(wav, dtype=np.float64), sampling_frequency=int(sr))
    out = parselmouth.praat.call(snd, 'Change gender', 75.0, 600.0, float(ratio), 0.0, 1.0, 1.0)
    return out.values[0].astype(np.float32)


def tune_audio(wav, sr, params):
    """按参数依次调音, 返回 (wav, sr)"""
    # 变速 (保持音高)
    ts = params.get('time_stretch', 1.0)
    if ts and ts != 1.0:
        wav = librosa.effects.time_stretch(wav, rate=ts)
    # 变调 (保持共振峰, 保持时长)
    ps = params.get('pitch_shift', 0)
    if ps:
        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=ps)
    # 共振峰移位 (Praat, 保持音高/时长)
    fs = params.get('formant_shift', 1.0)
    if fs and fs != 1.0:
        wav = _formant_shift(wav, sr, fs)
    # 高通 (去低频)
    hp = params.get('highpass', 0)
    if hp and hp > 0:
        b, a = _butter_highpass(hp, sr)
        wav = lfilter(b, a, wav)
    # 低通 (去高频)
    lp = params.get('lowpass', 0)
    if lp and lp > 0:
        b, a = _butter_lowpass(lp, sr)
        wav = lfilter(b, a, wav)
    # 增益
    gain_db = params.get('gain_db', 0)
    if gain_db:
        wav = wav * (10 ** (gain_db / 20.0))
    # 混响
    rv = params.get('reverb', 0)
    if rv:
        wav = _apply_reverb(np.asarray(wav, dtype=np.float32), sr, rv)
    # 归一化
    if params.get('normalize', True):
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * (10 ** (-1 / 20.0))  # -1dBFS
    return wav.astype(np.float32)


def main():
    if len(sys.argv) < 3:
        print('用法: audio_tune.py <输入音频> <参数JSON> [输出路径]', flush=True)
        sys.exit(1)
    in_path = sys.argv[1]
    try:
        params = json.loads(sys.argv[2])
    except Exception:
        print('[error] 参数JSON解析失败', flush=True)
        sys.exit(1)
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(in_path)[0] + '_tuned.wav'

    t0 = time.perf_counter()
    wav, sr = librosa.load(in_path, sr=None, mono=True)
    print(f'[tune] 载入 {os.path.basename(in_path)}: {len(wav)/sr:.1f}s @ {sr}Hz', flush=True)
    wav = tune_audio(wav, sr, params)
    target_sr = params.get('target_sr', sr)
    if target_sr and target_sr != sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    sf.write(out_path, wav, sr)
    print(f'[tune] done {time.perf_counter()-t0:.1f}s -> {out_path}', flush=True)
    print(f'RESULT:{out_path}', flush=True)


if __name__ == '__main__':
    main()
