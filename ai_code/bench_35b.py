# -*- coding: utf-8 -*-
"""35b (llama.cpp) 输出 token 详细测速
测: 首token延迟 / 生成总耗时 / think+content token 数 / 有效 tok/s / 总 tok/s
用非流式 (35b 流式 content 为空, 非流式才是真实可用速度)
"""
import urllib.request, json, sys, time
sys.stdout.reconfigure(encoding='utf-8')

URL = 'http://127.0.0.1:8080/v1/chat/completions'

def gen(messages, max_tokens):
    body = json.dumps({'model': 'qwen', 'messages': messages, 'max_tokens': max_tokens, 'stream': False}).encode('utf-8')
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'}), timeout=300)
    d = json.loads(r.read())
    dt = time.time() - t0
    msg = d['choices'][0]['message']
    return {
        'wall_s': dt,
        'think': msg.get('reasoning_content', ''),
        'content': msg.get('content', ''),
        'usage': d.get('usage', {}),
    }

def report(name, max_tokens, prompt):
    print(f'\n=== {name} (max_tokens={max_tokens}) ===', flush=True)
    t0 = time.time()
    r = gen([{'role': 'user', 'content': prompt}], max_tokens)
    prep_s = time.time() - t0 - r['wall_s']  # 请求搭建耗时(近似0)
    wall = r['wall_s']
    n_think = len(r['think'])
    n_content = len(r['content'])
    n_total = n_think + n_content
    usage = r['usage']
    print(f'  墙钟总耗时: {wall:.1f}s', flush=True)
    print(f'  think 字符数: {n_think} ({n_think/4:.0f} token估)', flush=True)
    print(f'  content 字符数: {n_content} ({n_content/4:.0f} token估)', flush=True)
    print(f'  usage: prompt_tokens={usage.get("prompt_tokens")} completion_tokens={usage.get("completion_tokens")} total={usage.get("total_tokens")}', flush=True)
    ct = usage.get('completion_tokens', n_total)
    if ct and wall > 0:
        print(f'  **生成速度: {ct/wall:.2f} tok/s (completion_tokens/墙钟)**', flush=True)
    # 首token延迟: 非流式拿不到, 但可通过 max_tokens=1 估
    print(f'  输出预览: {r["content"][:60]!r}', flush=True)
    return {'wall': wall, 'ct': ct, 'think': n_think, 'content': n_content}

# 1. 首 token 延迟估算 (max_tokens=1, 看模型多快出第一个completion token)
print('--- 测首token延迟 (max_tokens=1) ---', flush=True)
t0=time.time()
r1 = gen([{'role':'user','content':'你好'}], 1)
dt1 = time.time()-t0
print(f'  max_tokens=1 请求: {dt1:.2f}s (含思考启动)', flush=True)
print(f'  content={r1["content"]!r} think_len={len(r1["think"])}', flush=True)

# 2. 短输出
report('短对话', 100, '用一句话回答: 1+1等于几?')
# 3. 中输出
report('中对话', 300, '请详细解释一下什么是机器学习, 300字左右')
# 4. 长输出 (能跑则跑)
try:
    report('长输出', 800, '请写一篇关于人工智能未来的短文, 800字左右')
except Exception as e:
    print(f'  长输出失败: {str(e)[:100]}', flush=True)
print('\nDONE', flush=True)
