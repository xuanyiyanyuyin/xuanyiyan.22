"""全局日志: 实时记录所有输入输出, 只写不删"""
import os, time, sys, atexit

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class SessionLog:
    def __init__(self, name="session"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(LOG_DIR, f"{name}_{ts}.log")
        self.f = open(self.path, "w", encoding="utf-8", buffering=1)  # line-buffered
        self._write(f"=== {name} START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        atexit.register(self.close)

    def _write(self, text):
        self.f.write(text); self.f.flush()

    def log(self, level, msg):
        ts = time.strftime("%H:%M:%S")
        self._write(f"[{ts}] [{level}] {msg}\n")

    def info(self, msg): self.log("INFO", msg)
    def metric(self, key, value): self.log("METRIC", f"{key}={value}")
    def epoch(self, ep, **metrics):
        items = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self._write(f"[EPOCH {ep:4d}] {items}\n")
    def batch(self, bi, nb, **metrics):
        items = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self._write(f"  [BATCH {bi+1}/{nb}] {items}\n")
    def error(self, msg): self.log("ERROR", msg)
    def close(self):
        if not self.f.closed:
            self._write(f"=== END {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            self.f.close()

# Global singleton
_log = None

def get_log(name=None):
    global _log
    if _log is None or _log.f.closed:
        script = os.path.splitext(os.path.basename(sys.argv[0]))[0] if sys.argv else "run"
        _log = SessionLog(name or script)
    return _log

def info(msg): get_log().info(msg)
def metric(k,v): get_log().metric(k,v)
def epoch(ep,**kw): get_log().epoch(ep,**kw)
def batch(bi,nb,**kw): get_log().batch(bi,nb,**kw)
def error(msg): get_log().error(msg)
