# -*- coding: utf-8 -*-
"""35b 给足 max_tokens 测真实正文速度 (非流式)
- 拿 usage 的 completion_tokens 分 think/content 长度
- 但非流式只给总 completion_tokens, 分不出 think/content 各多少
- 所以额外用流式(即使content空, reasoning会流式) + 一次大token非流式对比
"""
import urllib.request, json, sys, time
sys.stdout.reconfigure(encoding='utf-8')

URL = 'http://127.0.0.1:8080/v1/chat/completions'

def gen(messages, max_tokens):
    body = json.dumps({'model': 'qwen', 'messages': messages, 'max_tokens': max_tokens, 'stream': False}).encode('utf-8')
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'}), timeout=600)
    d = json.loads(r.read())
    dt = time.time() - t0
    msg = d['choices'][0]['message']
    return dt, msg, d.get('usage', {})

# 给足 max_tokens, 让正文出来
for mt in (1500, 2000):
    print(f'\n=== max_tokens={mt} ===', flush=True)
    t0 = time.time()
    try:
        dt, msg, usage = gen([{'role':'user','content':'请写一篇500字的短文, 介绍你自己'}], mt)
        think = msg.get('reasoning_content','')
        content = msg.get('content','')
        ct = usage.get('completion_tokens', 0)
        print(f'  墙钟: {dt:.1f}s', flush=True)
        print(f'  think字符: {len(think)} ({len(think)/4:.0f} tok估)', flush=True)
        print(f'  content字符: {len(content)} ({len(content)/4:.0f} tok估)', flush=True)
        print(f'  usage: prompt={usage.get("prompt_tokens")} completion={ct} total={usage.get("total_tokens")}', flush=True)
        if ct and dt>0:
            print(f'  **总生成: {ct/dt:.2f} tok/s**', flush=True)
        print(f'  正文内容: {content[:100]!r}', flush=True)
        if content:
            # 估算正文阶段速度: 假设思考约 400-800 token, 正文=ct-思考
            for think_tok in (400, 600, 800):
                content_tok = max(0, ct - think_tok)
                if content_tok:
                    print(f'    (若思考={think_tok}token) 正文={content_tok}token → {content_tok/dt:.1f} tok/s', flush=True)
            break  # 成功出正文就停
    except Exception as e:
        print(f'  失败: {str(e)[:120]}', flush=True)
        break
print('\nDONE', flush=True)
