# -*- coding: utf-8 -*-
"""[2026-10-06] 鼠标物理移动记录 · Raw Input 版（WM_INPUT，拿原始 counts）
物理距离 = counts / DPI。默认 DPI=26000（主人鼠标）。8000Hz 回报也不丢事件（事件推送，非采样）。
用法: python mouse_raw.py [--dpi 26000] [--out DIR]   结束 Ctrl+C 自动写汇总
记录: 每包 counts(dx,dy) / 物理距离(cm) / 瞬时速度(cm/s) / 左右中键 / 急停与方向反转次数"""
import argparse, ctypes, ctypes.wintypes as w, datetime, json, math, os, sys, time
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
U = ctypes.windll.user32
# [2026-10-06] 声明 64 位签名：不声明时 lParam 指针会 OverflowError（int too long to convert）
U.DefWindowProcW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
U.DefWindowProcW.restype = ctypes.c_ssize_t
U.GetRawInputData.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(w.UINT), ctypes.c_uint]
U.GetRawInputData.restype = ctypes.c_uint
WM_INPUT = 0x00FF; RID_INPUT = 0x10000003; RIDEV_INPUTSINK = 0x00000100
class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", w.USHORT), ("usUsage", w.USHORT), ("dwFlags", w.DWORD), ("hwndTarget", w.HWND)]
class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", w.DWORD), ("dwSize", w.DWORD), ("hDevice", w.HANDLE), ("wParam", w.WPARAM)]
class RAWMOUSE(ctypes.Structure):
    _fields_ = [("usFlags", w.USHORT), ("ulButtons", w.ULONG), ("usButtonFlags", w.USHORT),
                ("usButtonData", w.USHORT), ("ulRawButtons", w.ULONG), ("lLastX", ctypes.c_long),
                ("lLastY", ctypes.c_long), ("ulExtraInformation", w.ULONG)]
class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]
class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", w.UINT), ("lpfnWndProc", ctypes.c_void_p), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", w.HANDLE), ("hIcon", w.HANDLE),
                ("hCursor", w.HANDLE), ("hbrBackground", w.HANDLE), ("lpszMenuName", w.LPCWSTR),
                ("lpszClassName", w.LPCWSTR)]
S = {"t0": time.perf_counter(), "start_iso": datetime.datetime.now().isoformat(timespec="milliseconds"),
     "cnt": 0, "pkts": 0, "clicks": {"left":0,"right":0,"middle":0,"x1":0,"x2":0}, "per_min": {},
     "corr": 0, "corr_t": [], "stops": 0, "dir": None, "last_move_t": None, "speeds": [],
     "peak": 0.0, "buttons": set(), "last_pt_t": None}
FH = []; ARGS = [None]
def iso_ms(): return datetime.datetime.now().strftime("%H:%M:%S.")+("%03d"%(datetime.datetime.now().microsecond//1000))
def log(o):
    o["_ms"] = int((time.perf_counter()-S["t0"])*1000); o["_wall"] = iso_ms()
    if FH: FH[0].write(json.dumps(o, ensure_ascii=False)+chr(10)); FH[0].flush()
def flush(out):
    os.makedirs(out, exist_ok=True); d = datetime.datetime.now().strftime("%Y%m%d")
    dpi = ARGS[0].dpi; cm = S["cnt"] / dpi * 2.54
    sp = S["speeds"]; ct = S["corr_t"]
    cg = [round(ct[i]-ct[i-1], 2) for i in range(1, len(ct))]
    dur = time.perf_counter() - S["t0"]
    s = {"date": d, "start_iso": S["start_iso"], "end_iso": datetime.datetime.now().isoformat(timespec="milliseconds"),
         "duration_s": round(dur, 1), "dpi": dpi, "packets": S["pkts"], "counts": S["cnt"],
         "physical_cm": round(cm, 1), "physical_m": round(cm/100, 2),
         "avg_cm_per_s": round(cm/dur, 2) if dur else 0, "peak_cm_per_s": round(S["peak"], 2),
         "clicks": S["clicks"], "corrections": S["corr"], "avg_corr_gap_s": round(sum(cg)/len(cg), 2) if cg else 0,
         "stops": S["stops"], "per_min_packets": S["per_min"]}
    fp = os.path.join(out, "mouse_summary_%s.json" % d)
    open(fp, "w", encoding="utf-8").write(json.dumps(s, ensure_ascii=False, indent=1))
    print(chr(10)+"[汇总] "+fp)
    print("物理移动: %.1f cm (%.2f m) | 平均 %.2f cm/s | 峰值 %.2f cm/s" % (cm, cm/100, s["avg_cm_per_s"], S["peak"]))
    print("按键: %s | 方向反转 %d 次 | 急停 %d 次" % (S["clicks"], S["corr"], S["stops"]))
def wndproc(hwnd, msg, wp, lp):
    if msg == WM_INPUT:
        size = w.UINT(0)
        U.GetRawInputData(wp, RID_INPUT, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
        buf = ctypes.create_string_buffer(size.value)
        if U.GetRawInputData(wp, RID_INPUT, buf, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER)) == size.value:
            ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
            if ri.header.dwType == 0:   # RIM_TYPEMOUSE
                dx, dy = ri.mouse.lLastX, ri.mouse.lLastY
                now = time.perf_counter()
                if dx or dy:
                    dist = math.hypot(dx, dy); S["cnt"] += int(dist); S["pkts"] += 1
                    mi = int((now-S["t0"])/60); S["per_min"][mi] = S["per_min"].get(mi, 0)+1
                    if S["last_pt_t"] is not None:
                        dtms = (now-S["last_pt_t"])*1000.0
                        if dtms > 0:
                            v = dist / ARGS[0].dpi * 2.54 / (dtms/1000.0)
                            S["speeds"].append(round(v, 1))
                            if v > S["peak"]: S["peak"] = v
                            if v < 3.0: S["stops"] += 1
                    S["last_pt_t"] = now
                    if abs(dx) > 1:
                        dd = 1 if dx > 0 else -1
                        if S["dir"] is not None and dd != S["dir"] and dist > 10:
                            S["corr"] += 1; S["corr_t"].append(now)
                        S["dir"] = dd
                    log({"type":"move","dx":dx,"dy":dy,"counts":int(dist),"cm":round(dist/ARGS[0].dpi*2.54, 4)})
                bf = ri.mouse.usButtonFlags
                for flag, name in ((0x0001,"left"),(0x0002,"right"),(0x0004,"middle"),(0x0040,"x1"),(0x0080,"x2")):
                    if bf & flag:
                        S["clicks"][name] += 1
                        log({"type":"click","button":name})
                if bf & 0x0400: log({"type":"wheel","delta": ctypes.c_short(ri.mouse.usButtonData).value})
        return 0
    if msg == 0x0002:  # WM_DESTROY
        U.PostQuitMessage(0); return 0
    return U.DefWindowProcW(hwnd, msg, wp, lp)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dpi", type=float, default=26000.0)
    ap.add_argument("--out", default=r"D:\\ai\\input_stats"); a = ap.parse_args(); ARGS[0] = a
    os.makedirs(a.out, exist_ok=True)
    FH.append(open(os.path.join(a.out, "mouse_raw_%s.jsonl" % datetime.datetime.now().strftime("%Y%m%d")), "a", encoding="utf-8"))
    print("[启动] Raw Input 记录中 | DPI=%.0f | 输出: %s (Ctrl+C 结束)" % (a.dpi, a.out), flush=True)
    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
    proc = WNDPROC(wndproc)
    hinst = ctypes.windll.kernel32.GetModuleHandleW(None)
    cls = WNDCLASSW(); cls.lpfnWndProc = ctypes.cast(proc, ctypes.c_void_p); cls.hInstance = hinst; cls.lpszClassName = "RawMouseMon"
    U.RegisterClassW(ctypes.byref(cls))
    hwnd = U.CreateWindowExW(0, "RawMouseMon", "raw", 0, 0, 0, 0, 0, None, None, hinst, None)
    if not hwnd: print("[错误] 建窗口失败"); return 1
    rid = RAWINPUTDEVICE(1, 2, RIDEV_INPUTSINK, hwnd)
    if not U.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(RAWINPUTDEVICE)):
        print("[错误] RegisterRawInputDevices 失败"); return 1
    print("[就绪] 现在可以打游戏了（本窗口最小化也行）", flush=True)
    lf = time.time(); msg = w.MSG()
    try:
        while U.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            U.TranslateMessage(ctypes.byref(msg)); U.DispatchMessageW(ctypes.byref(msg))
            if time.time()-lf > 60: lf = time.time(); flush(a.out)
    except KeyboardInterrupt: pass
    finally:
        if FH: FH[0].close()
        flush(a.out)
    return 0
if __name__ == "__main__": sys.exit(main())
