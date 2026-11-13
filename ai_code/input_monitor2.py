# -*- coding: utf-8 -*-
"""[2026-10-06] 输入行为记录 · 轮询版（不用钩子 ⇒ 不受杀软/权限限制）
键盘: GetAsyncKeyState 200Hz 采样; 鼠标: GetCursorPos + 左右中键状态
用法: python input_monitor2.py [--hz 200] [--out DIR]   结束 Ctrl+C 自动写汇总"""
import argparse, ctypes, ctypes.wintypes as w, datetime, json, math, os, sys, time
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
U = ctypes.windll.user32
# [2026-10-06] 物理换算: 屏幕 px -> counts -> cm（DPI 26000；Windows 指针速度档见下表）
MULT = {1:1/32.0, 2:1/16.0, 3:1/8.0, 4:1/4.0, 5:1/2.0, 6:3/4.0, 7:1.0, 8:1.25, 9:1.5, 10:1.75, 11:2.0,
        12:2.25, 13:2.5, 14:2.75, 15:3.0, 16:3.25, 17:3.5, 18:3.75, 19:4.0, 20:4.5}
def _sys_mouse():
    import ctypes as _c
    sp = _c.c_int(10); arr = (_c.c_int*3)(0,0,0)
    try:
        _c.windll.user32.SystemParametersInfoW(0x0070, 0, _c.byref(sp), 0)
        _c.windll.user32.SystemParametersInfoW(0x0003, 0, _c.byref(arr), 0)
    except Exception: pass
    return sp.value or 10, bool(arr[2])
SPEED, ACCEL = _sys_mouse()
MULT_V = MULT.get(SPEED, 1.0)
def px_to_cm(px, dpi=26000.0):
    counts = px / MULT_V          # 关加速时: counts = px / 倍率
    return counts / dpi * 2.54
VK = {0x08:"Backspace",0x09:"Tab",0x0D:"Enter",0x10:"Shift",0x11:"Ctrl",0x12:"Alt",0x14:"CapsLock",0x1B:"Esc",
      0x20:"Space",0x21:"PgUp",0x22:"PgDn",0x23:"End",0x24:"Home",0x25:"Left",0x26:"Up",0x27:"Right",
      0x28:"Down",0x2D:"Insert",0x2E:"Delete",0x5B:"Win",0xBA:";",0xBB:"=",0xBC:",",0xBD:"-",0xBE:".",
      0xBF:"/",0xC0:"`",0xDB:"[",0xDC:"\\",0xDD:"]",0xDE:"'"}
for _i in range(0x30,0x3A): VK[_i]=chr(_i)
for _i in range(0x41,0x5B): VK[_i]=chr(_i)
for _i in range(0x70,0x7C): VK[_i]="F%d"%(_i-0x6F)
VKS = sorted(VK.keys())
LB, RB, MB = 0x01, 0x02, 0x04
S = {"keys":{}, "clicks":{"left":0,"right":0,"middle":0}, "gap":[], "burst":0, "px":0.0,
     "bb":[10**9,10**9,-10**9,-10**9], "moves":0, "pt":None, "dir":None, "corr":0, "corr_t":[], "n":0,
     "last":None, "t0":time.perf_counter(), "start_iso":datetime.datetime.now().isoformat(timespec="milliseconds"),
     "down":set(), "btn":set(), "per_min":{}}
FH=[]
def iso_ms(): return datetime.datetime.now().strftime("%H:%M:%S.")+("%03d"%(datetime.datetime.now().microsecond//1000))
def log(o):
    o["_ms"]=int((time.perf_counter()-S["t0"])*1000); o["_wall"]=iso_ms()
    if FH: FH[0].write(json.dumps(o,ensure_ascii=False)+chr(10)); FH[0].flush()
def flush(out):
    os.makedirs(out,exist_ok=True); d=datetime.datetime.now().strftime("%Y%m%d"); g=S["gap"]; ct=S["corr_t"]
    cg=[round(ct[i]-ct[i-1],2) for i in range(1,len(ct))]
    s={"date":d,"start_iso":S["start_iso"],"end_iso":datetime.datetime.now().isoformat(timespec="milliseconds"),
       "duration_s":round(time.perf_counter()-S["t0"],1),"events":S["n"],
       "keyboard":{"presses":sum(S["keys"].values()),"distinct":len(S["keys"]),
         "top":sorted(S["keys"].items(),key=lambda x:-x[1])[:20],
         "avg_gap_ms":round(sum(g)/len(g),1) if g else 0,
         "median_gap_ms":round(sorted(g)[len(g)//2],1) if g else 0,
         "bursts_lt200ms":S["burst"],"peak_per_min":max(S["per_min"].values()) if S["per_min"] else 0},
       "physical":{"dpi":26000,"win_speed":SPEED,"accel":ACCEL,"mult":round(MULT_V,4),"total_cm":round(px_to_cm(S["px"]),1),"avg_cm_s":round(px_to_cm(S["px"])/max(1,(time.perf_counter()-S["t0"])),2)},
       "mouse":{"left":S["clicks"]["left"],"right":S["clicks"]["right"],"middle":S["clicks"]["middle"],
         "move_px":round(S["px"]),"move_bbox":(S["bb"] if S["bb"][0]<10**8 else None),
         "bbox_size":([S["bb"][2]-S["bb"][0],S["bb"][3]-S["bb"][1]] if S["bb"][0]<10**8 else None),
         "samples":S["moves"],"corrections":S["corr"],"avg_corr_gap_s":round(sum(cg)/len(cg),2) if cg else 0},
       "per_min":S["per_min"]}
    fp=os.path.join(out,"summary_%s.json"%d)
    open(fp,"w",encoding="utf-8").write(json.dumps(s,ensure_ascii=False,indent=1))
    print(chr(10)+"[汇总] "+fp); print(json.dumps({k:s[k] for k in ("duration_s","events","keyboard","mouse")},ensure_ascii=False,indent=1)[:1200])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=r"D:\\ai\\input_stats")
    ap.add_argument("--hz",type=int,default=200); a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    FH.append(open(os.path.join(a.out,"raw_%s.jsonl"%datetime.datetime.now().strftime("%Y%m%d")),"a",encoding="utf-8"))
    dt=1.0/max(20,a.hz); print("[启动] 轮询记录中 hz=%d 输出: %s  (Ctrl+C 结束并写汇总)"%(a.hz,a.out),flush=True)
    pt=w.POINT(); last_flush=time.time(); n=0
    try:
        while True:
            now_iso=None
            for vk in VKS:
                down = bool(U.GetAsyncKeyState(vk) & 0x8000)
                if down and vk not in S["down"]:
                    S["down"].add(vk); nm=VK.get(vk,"VK_%02X"%vk); S["keys"][nm]=S["keys"].get(nm,0)+1; S["n"]+=1
                    t=time.perf_counter()
                    if S["last"]:
                        gp=(t-S["last"])*1000.0; S["gap"].append(gp)
                        if gp<200: S["burst"]+=1
                    S["last"]=t
                    mi=int((t-S["t0"])/60); S["per_min"][mi]=S["per_min"].get(mi,0)+1
                    log({"type":"key","key":nm})
                elif not down and vk in S["down"]:
                    S["down"].discard(vk)
            U.GetCursorPos(ctypes.byref(pt)); x,y=pt.x,pt.y; S["moves"]+=1
            S["bb"][0]=min(S["bb"][0],x); S["bb"][1]=min(S["bb"][1],y)
            S["bb"][2]=max(S["bb"][2],x); S["bb"][3]=max(S["bb"][3],y)
            if S["pt"]:
                dx,dy=x-S["pt"][0],y-S["pt"][1]; d=math.hypot(dx,dy); S["px"]+=d
                if abs(dx)>3:
                    dd=1 if dx>0 else -1
                    if S["dir"] is not None and dd!=S["dir"] and d>20:
                        S["corr"]+=1; S["corr_t"].append(time.perf_counter())
                    S["dir"]=dd
            S["pt"]=(x,y)
            # [2026-10-06] 把移动也写进 raw（每 5 次采样≈25ms 一次），便于事后分析轨迹/修正
            _mv = S.get("mv", 0) + 1; S["mv"] = _mv
            if _mv % 5 == 0:
                log({"type":"move","x":x,"y":y})
            for vk,btn in ((LB,"left"),(RB,"right"),(MB,"middle")):
                dn = bool(U.GetAsyncKeyState(vk) & 0x8000)
                if dn and vk not in S["btn"]:
                    S["btn"].add(vk); S["clicks"][btn]+=1; S["n"]+=1
                    mi=int((time.perf_counter()-S["t0"])/60); S["per_min"][mi]=S["per_min"].get(mi,0)+1
                    log({"type":"click","button":btn,"x":x,"y":y})
                elif not dn and vk in S["btn"]:
                    S["btn"].discard(vk)
            if time.time()-last_flush>60: last_flush=time.time(); flush(a.out)
            time.sleep(dt)
    except KeyboardInterrupt: pass
    finally:
        if FH: FH[0].close()
        flush(a.out)
    return 0
if __name__ == "__main__":
    import atexit
    atexit.register(lambda: (FH and FH[0].close(), flush(sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else r"D:\\ai\\input_stats")))
    sys.exit(main())
