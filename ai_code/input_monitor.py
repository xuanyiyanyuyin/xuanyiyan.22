# -*- coding: utf-8 -*-
"""[2026-10-06] 本机输入行为记录（纯标准库 ctypes，无需安装、无需管理员）
键盘: 按键名/间隔/次数/连击;  鼠标: 左右中键次数/移动距离/范围bbox/方向修正次数与间隔
用法: python input_monitor.py [--count-only] [--out DIR]   结束按 Ctrl+C 自动写汇总
仅用于记录本机自己的操作习惯，勿用于他人设备。"""
import argparse, ctypes, ctypes.wintypes as w, datetime, json, math, os, sys, time
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
U = ctypes.windll.user32; K = ctypes.windll.kernel32
WH_KEYBOARD_LL, WH_MOUSE_LL = 13, 14
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_RBUTTONDOWN, WM_MBUTTONDOWN = 0x0200, 0x0201, 0x0204, 0x0207
VK = {0x08:"Backspace",0x09:"Tab",0x0D:"Enter",0x10:"Shift",0x11:"Ctrl",0x12:"Alt",0x14:"CapsLock",
      0x1B:"Esc",0x20:"Space",0x21:"PgUp",0x22:"PgDn",0x23:"End",0x24:"Home",0x25:"Left",0x26:"Up",
      0x27:"Right",0x28:"Down",0x2D:"Insert",0x2E:"Delete",0x5B:"Win",0xBA:";",0xBB:"=",0xBC:",",
      0xBD:"-",0xBE:".",0xBF:"/",0xC0:"`",0xDB:"[",0xDC:"\\",0xDD:"]",0xDE:"'"}
for _i in range(0x30,0x3A): VK[_i]=chr(_i)
for _i in range(0x41,0x5B): VK[_i]=chr(_i)
for _i in range(0x70,0x7C): VK[_i]="F%d"%(_i-0x6F)
class KBD(ctypes.Structure):
    _fields_=[("vkCode",w.DWORD),("scanCode",w.DWORD),("flags",w.DWORD),("time",w.DWORD),("extra",ctypes.c_void_p)]
class MS(ctypes.Structure):
    _fields_=[("pt",w.POINT),("mouseData",w.DWORD),("flags",w.DWORD),("time",w.DWORD),("extra",ctypes.c_void_p)]
S={"start":time.time(),"t0":time.perf_counter(),"start_iso":datetime.datetime.now().isoformat(timespec="milliseconds"),"keys":{},"clicks":{"left":0,"right":0,"middle":0},"gap":[],"burst":0,
   "px":0.0,"bb":[10**9,10**9,-10**9,-10**9],"moves":0,"pt":None,"dir":None,"corr":0,"corr_t":[],"n":0}
FH=[None]; ARGS=[None]
def log(o):
    try:
        o["_ms"]=int((time.perf_counter()-S["t0"])*1000)          # 相对开始毫秒(高精度)
        o["_wall"]=datetime.datetime.now().strftime("%H:%M:%S.")+("%03d"%(datetime.datetime.now().microsecond//1000))
    except Exception:
        pass
    if FH[0]:
        FH[0].write(json.dumps(o,ensure_ascii=False)+chr(10)); FH[0].flush()
def flush(out):
    os.makedirs(out,exist_ok=True)
    d=datetime.datetime.now().strftime("%Y%m%d"); g=S["gap"]; ct=S["corr_t"]
    cg=[round(ct[i]-ct[i-1],2) for i in range(1,len(ct))]
    s={"date":d,"start_iso":S.get("start_iso"),"end_iso":datetime.datetime.now().isoformat(timespec="milliseconds"),"duration_s":round(time.time()-S["start"],1),"events":S["n"],
       "keyboard":{"presses":sum(S["keys"].values()),"distinct":len(S["keys"]),
         "top":sorted(S["keys"].items(),key=lambda x:-x[1])[:20],
         "avg_gap_ms":round(sum(g)/len(g),1) if g else 0,
       "median_gap_ms":round(sorted(g)[len(g)//2],1) if g else 0,"bursts_lt200ms":S["burst"]},
       "mouse":{"left":S["clicks"]["left"],"right":S["clicks"]["right"],"middle":S["clicks"]["middle"],
         "move_px":round(S["px"]),"move_bbox":(S["bb"] if S["bb"][0]<10**8 else None),
       "bbox_size":([S["bb"][2]-S["bb"][0],S["bb"][3]-S["bb"][1]] if S["bb"][0]<10**8 else None),
         "samples":S["moves"],"corrections":S["corr"],"avg_corr_gap_s":round(sum(cg)/len(cg),2) if cg else 0}}
    fp=os.path.join(out,"summary_%s.json"%d)
    open(fp,"w",encoding="utf-8").write(json.dumps(s,ensure_ascii=False,indent=1))
    print(chr(10)+"[汇总] "+fp); print(json.dumps(s,ensure_ascii=False,indent=1)[:1000])
def kproc(nc,wp,lp):
    if nc==0 and wp in (WM_KEYDOWN,WM_SYSKEYDOWN):
        k=ctypes.cast(lp,ctypes.POINTER(KBD)).contents; now=time.time()
        nm=VK.get(k.vkCode,"VK_%02X"%k.vkCode); S["keys"][nm]=S["keys"].get(nm,0)+1; S["n"]+=1
        if S.get("last"):
            gp=(now-S["last"])*1000.0; S["gap"].append(gp)
            if gp<200: S["burst"]+=1
        S["last"]=now
        log({"t":round(now,3),"type":"key","key":("*" if ARGS[0].count_only else nm)})
    return U.CallNextHookEx(None,nc,wp,lp)
def mproc(nc,wp,lp):
    if nc==0:
        m=ctypes.cast(lp,ctypes.POINTER(MS)).contents; now=time.time(); x,y=m.pt.x,m.pt.y
        if wp==WM_MOUSEMOVE:
            S["moves"]+=1
            S["bb"][0]=min(S["bb"][0],x); S["bb"][1]=min(S["bb"][1],y)
            S["bb"][2]=max(S["bb"][2],x); S["bb"][3]=max(S["bb"][3],y)
            if S["pt"]:
                dx,dy=x-S["pt"][0],y-S["pt"][1]; d=math.hypot(dx,dy); S["px"]+=d
                if abs(dx)>3:
                    dd=1 if dx>0 else -1
                    if S["dir"] is not None and dd!=S["dir"] and d>20:
                        S["corr"]+=1; S["corr_t"].append(now)
                    S["dir"]=dd
            S["pt"]=(x,y)
        elif wp in (WM_LBUTTONDOWN,WM_RBUTTONDOWN,WM_MBUTTONDOWN):
            b="left" if wp==WM_LBUTTONDOWN else ("right" if wp==WM_RBUTTONDOWN else "middle")
            S["clicks"][b]+=1; S["n"]+=1
            log({"t":round(now,3),"type":"click","button":b,"x":x,"y":y})
    return U.CallNextHookEx(None,nc,wp,lp)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=r"D:\\ai\\input_stats")
    ap.add_argument("--count-only",action="store_true"); a=ap.parse_args(); ARGS[0]=a
    os.makedirs(a.out,exist_ok=True)
    if not a.count_only:
        FH[0]=open(os.path.join(a.out,"raw_%s.jsonl"%datetime.datetime.now().strftime("%Y%m%d")),"a",encoding="utf-8")
    print("[启动] 记录中… 输出: "+a.out+"  (Ctrl+C 结束并写汇总)")
    if a.count_only: print("       隐私模式: 只统计次数，不写键名")
    KP=ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.c_int,w.WPARAM,w.LPARAM)(kproc)
    MP=ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.c_int,w.WPARAM,w.LPARAM)(mproc)
    hk=U.SetWindowsHookExW(WH_KEYBOARD_LL,KP,K.GetModuleHandleW(None),0)
    hm=U.SetWindowsHookExW(WH_MOUSE_LL,MP,K.GetModuleHandleW(None),0)
    if not hk or not hm: print("[错误] 钩子安装失败（可能被杀软阻止）"); return 1
    lf=time.time()
    try:
        msg=w.MSG()
        while U.GetMessageW(ctypes.byref(msg),None,0,0)!=0:
            if time.time()-lf>60: lf=time.time(); flush(a.out)
    except KeyboardInterrupt: pass
    finally:
        U.UnhookWindowsHookEx(hk); U.UnhookWindowsHookEx(hm)
        if FH[0]: FH[0].close()
        flush(a.out)
    return 0
def _bye():
    """[2026-10-06] 直接关窗口(Ctrl+C 之外的方式)时也写汇总，防丢"""
    try:
        if ARGS[0] is not None:
            if FH[0]:
                FH[0].close()
            flush(ARGS[0].out)
    except Exception:
        pass


if __name__ == "__main__":
    import atexit
    atexit.register(_bye)
    sys.exit(main())
