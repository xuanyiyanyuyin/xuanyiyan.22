"""H3-Image 原生权重 -> diffusers 布局转换包装器（只写未跑，点火需主人下令）。

官方脚本 scripts/convert_minimax_h3_to_diffusers.py 不在权重仓里，
本机出网受限，这里在点火时自动拉取（直连 -> 代理 -> 镜像兜底）。
转换会复制 63G 文本编码器，总写盘约 131G，先查盘空间。
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

NATIVE = r'D:\ai\models\MiniMax-H3-Image'
ENCODER = r'D:\ai\models\MiniMax-H3\Ref2VA\text_encoder'
OUT = r'D:\ai\models\MiniMax-H3-Image-diffusers'
SCRIPT = r'D:\ai\convert_minimax_h3_to_diffusers.py'
URLS = [
    'https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_minimax_h3_to_diffusers.py',
    'https://ghfast.top/https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_minimax_h3_to_diffusers.py',
    'https://gh-proxy.com/https://raw.githubusercontent.com/huggingface/diffusers/main/scripts/convert_minimax_h3_to_diffusers.py',
]
NEED_GB = 140


def fetch_script():
    if os.path.isfile(SCRIPT) and os.path.getsize(SCRIPT) > 10000:
        print(f'[script] 已在盘: {SCRIPT}', flush=True)
        return
    proxies = [{'https': 'http://127.0.0.1:7892', 'http': 'http://127.0.0.1:7892'}, {}]
    for url in URLS:
        for px in proxies:
            try:
                op = urllib.request.build_opener(urllib.request.ProxyHandler(px))
                data = op.open(url, timeout=60).read()
                if b'argparse' in data and len(data) > 10000:
                    with open(SCRIPT, 'wb') as f:
                        f.write(data)
                    print(f'[script] 下载成功 {len(data)}B <- {url} (proxy={bool(px)})', flush=True)
                    return
            except Exception as e:
                print(f'[script] 失败 {url} proxy={bool(px)}: {type(e).__name__} {str(e)[:100]}', flush=True)
    sys.exit('RESULT: FAIL 官方转换脚本拉取失败，全部源不通；手动下载后放到 ' + SCRIPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print_only', action='store_true', help='只打印将执行的命令，不动盘')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    for p in (NATIVE, os.path.join(NATIVE, 'transformer'), ENCODER):
        if not os.path.isdir(p):
            sys.exit(f'RESULT: FAIL 缺路径: {p}')

    free_gb = shutil.disk_usage('D:\\').free / 1e9
    print(f'[disk] D 盘剩余 {free_gb:.1f} GB，需要约 {NEED_GB} GB', flush=True)
    if free_gb < NEED_GB:
        sys.exit(f'RESULT: FAIL 盘空间不足（{free_gb:.1f} < {NEED_GB}）')

    cmd = [sys.executable, SCRIPT,
           '--checkpoint_path', NATIVE,
           '--output_path', args.out,
           '--version', 'image',
           '--text_encoder_path', ENCODER]
    print('[cmd]', ' '.join(cmd), flush=True)
    if args.print_only:
        print('RESULT: PRINT_ONLY 未执行', flush=True)
        return

    fetch_script()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f'RESULT: FAIL 转换退出码 {r.returncode}')
    print(f'RESULT: DONE 转换完成 -> {args.out}', flush=True)


if __name__ == '__main__':
    main()