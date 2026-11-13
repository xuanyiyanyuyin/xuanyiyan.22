"""V4-Flash 对话版服务 — 提供与 qwen_server.py 相同接口 + OpenAI 兼容 HTTP (端口8123)
init_engine / generate_sync / reset_conversation / shutdown / ENGINE_READY
CherryStudio 连 http://127.0.0.1:8123/v1
"""
import torch, time, gc, json, os, http.server, sys
import v4flash_sched as vs
from transformers import AutoTokenizer
sys.path.insert(0, r'D:\ai\_v4flash_src')
from encoding_dsv4 import encode_messages, parse_message_from_completion_text
import v4flash_agent

ENGINE_READY = False
_engine = None
_tokenizer = None
_wrapper = None
_agent = None

# ═══ 读写头 (跨会话文件记忆: 字符级 TF-IDF + 余弦, 与 35B websearch_proxy 对齐) ═══
MEMORY_DIR = r'D:\ai\llamacpp\memory'  # 与 35B 共享同一记忆目录
os.makedirs(MEMORY_DIR, exist_ok=True)

def _tokenize(text):
    """中文友好: 字符级 1-2 gram (无需 jieba 分词)"""
    t = text.strip()
    grams = list(t)  # unigram
    grams += [t[i:i+2] for i in range(len(t)-1)]  # bigram
    return grams

def save_memory(user_msg, assistant_reply):
    """会话结束: 存一段对话到 .mem 文件 (JSON, 毫秒级文件名+防重名, 避免同秒覆盖)"""
    try:
        now = time.time()
        rec = {'ts': int(now), 'user': user_msg[:500], 'assistant': assistant_reply[:1500]}
        ms = int(now * 1000)
        fp = os.path.join(MEMORY_DIR, f'mem_{ms}.json')
        i = 1
        while os.path.exists(fp):
            fp = os.path.join(MEMORY_DIR, f'mem_{ms}_{i}.json')
            i += 1
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False)
    except Exception:
        pass

def recall_memory(current_msg, top_k=3, threshold=0.15):
    """新会话: 用当前消息检索最相似的历史记忆 (TF-IDF + 余弦)"""
    import glob
    files = sorted(glob.glob(os.path.join(MEMORY_DIR, '*.json')))
    if not files:
        return []
    recs = []
    for fp in files:
        try:
            with open(fp, encoding='utf-8') as f:
                recs.append(json.load(f))
        except Exception:
            continue
    if not recs:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    docs = [_tokenize(r['user']) for r in recs]
    vec = TfidfVectorizer(analyzer=lambda x: x, lowercase=False, token_pattern=None)
    try:
        tfidf = vec.fit_transform(docs)
        q = vec.transform([_tokenize(current_msg)])
        sims = cosine_similarity(q, tfidf)[0]
    except Exception:
        return []
    scored = sorted(zip(recs, sims), key=lambda x: -x[1])
    return [{'user': r['user'][:200], 'assistant': r['assistant'][:500], 'score': round(float(s), 3)}
            for r, s in scored if s >= threshold][:top_k]

# ═══ 联网搜索工具 (v4flash 无原生 function calling, 用提示词约定格式) ═══
TOOL_HINT = ('你可以使用 web_search 工具搜索互联网获取最新信息。'
             '当需要联网搜索时，请先只输出以下格式的工具调用，然后停止：\n'
             '<|tool_call|>\n{"name": "web_search", "arguments": {"query": "搜索关键词"}}\n</|tool_call|>')

def do_search(query):
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        return [{'title': r['title'], 'href': r['href'], 'body': r['body']} for r in results]
    except Exception as e:
        return [{'error': str(e)[:200]}]

def _safe_path(path, base=r'D:\ai'):
    p = os.path.abspath(path if os.path.isabs(path) else os.path.join(base, path))
    b = os.path.abspath(base)
    if p.lower() == b.lower() or p.lower().startswith(b.lower() + os.sep):
        return p
    return None

def system_status():
    import psutil
    st = {'time': time.strftime('%Y-%m-%d %H:%M:%S %A')}
    try:
        st['cpu_percent'] = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        st['memory'] = {'total_gb': round(mem.total / 1024**3, 1), 'used_gb': round(mem.used / 1024**3, 1), 'percent': mem.percent}
        st['disk_c_free_gb'] = round(psutil.disk_usage('C:\\').free / 1024**3, 1)
        st['disk_d_free_gb'] = round(psutil.disk_usage('D:\\').free / 1024**3, 1)
    except Exception as e:
        st['psutil_error'] = str(e)[:100]
    try:
        import subprocess
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout.strip()
        st['gpu'] = out if out else 'nvidia-smi 无输出'
    except Exception:
        st['gpu'] = 'nvidia-smi 不可用'
    return st

TEXT_EXTS = {'.txt', '.md', '.json', '.yml', '.yaml', '.py', '.bat', '.log', '.cfg', '.ini', '.sh', '.csv', '.toml'}

def read_local_file(path):
    p = _safe_path(path)
    if p is None:
        return {'error': '路径不在允许范围(仅限 D:/ai)'}
    if not os.path.isfile(p):
        return {'error': '文件不存在'}
    ext = os.path.splitext(p)[1].lower()
    if ext not in TEXT_EXTS:
        return {'error': '不支持该文件类型(%s), 只读文本类' % ext}
    if os.path.getsize(p) > 200 * 1024:
        return {'error': '文件太大, 只读 ≤200KB'}
    try:
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            return {'path': p, 'content': f.read()}
    except Exception as e:
        return {'error': str(e)[:200]}

def list_local_dir(path=''):
    p = _safe_path(path or '.')
    if p is None:
        return {'error': '路径不在允许范围(仅限 D:/ai)'}
    if not os.path.isdir(p):
        return {'error': '目录不存在'}
    items = []
    for name in os.listdir(p):
        full = os.path.join(p, name)
        try:
            if os.path.isdir(full):
                items.append({'name': name, 'type': 'dir'})
            else:
                items.append({'name': name, 'type': 'file', 'size': os.path.getsize(full)})
        except Exception:
            pass
    items.sort(key=lambda x: (x['type'] != 'dir', x['name'].lower()))
    return {'path': p, 'items': items[:80]}

def _extract_tool_call(text):
    import re
    m = re.search(r'<\|tool_call\|>\s*(\{.*?\})\s*</\|tool_call\|>', text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None

def _execute_tool(name, args):
    if isinstance(args, str):
        try:
            args = json.loads(args) if args else {}
        except Exception:
            args = {}
    args = args or {}
    if name == 'web_search':
        return do_search(args.get('query', ''))
    if name == 'get_time':
        return '当前系统时间: ' + time.strftime('%Y-%m-%d %H:%M:%S %A')
    if name == 'get_system_status':
        return system_status()
    if name == 'read_file':
        return read_local_file(args.get('path', ''))
    if name == 'list_dir':
        return list_local_dir(args.get('path', ''))
    return {'error': '未知工具: ' + str(name)}

def chat_with_tools(user_msg, max_new_tokens=1048576):
    """带联网搜索的对话循环 (最多2轮: 工具调用 -> 回喂 -> 最终回答)"""
    first = _wrapper.chat(f'{TOOL_HINT}\n\n{user_msg}', max_new_tokens)
    call = _extract_tool_call(first)
    if not call:
        return first
    if call.get('name') == 'web_search':
        q = (call.get('arguments') or {}).get('query', '')
        results = do_search(q)
        followup = (f'[web_search 工具结果] {json.dumps(results, ensure_ascii=False)[:6000]}\n\n'
                    '请基于以上搜索结果直接回答用户问题。')
        return _wrapper.chat(followup, max_new_tokens)
    return first


def chat_with_agent(user_msg, max_new_tokens=1048576):
    """完整工具集 agent 闭环 (文件读写删/代码执行/shell/联网/环境部署) — 独立入口, 不影响纯文本对话"""
    out = _agent.run(user_msg, max_new_tokens=max_new_tokens, do_sample=False)
    return out['text']


def chat_thinking(user_msg, max_new_tokens=1048576):
    """DSML thinking 对话: 深度推理 + 总结回答 (对齐 35B 深度推理机制)。
    返回 (reasoning_content, content)。"""
    msgs = [{'role': 'system', 'content': vs.CHAT_SYSTEM},
            {'role': 'user', 'content': user_msg}]
    prompt = encode_messages(msgs, thinking_mode='thinking')
    ids = _tokenizer.encode(prompt, add_special_tokens=False)
    ids_t = torch.tensor([ids], device=vs.DEVICE)
    with torch.inference_mode():
        # 采样对齐官方 chat_raw (DSML 格式): 无 top_k/top_p 截断, 无重复惩罚,
        # 否则 thinking 长推理下 EOS 会被 top_k=50 挤出, 模型永不停止 → 请求一直挂着
        out = _engine.generate(ids_t, max_new_tokens=max_new_tokens,
                               temperature=0.6, top_k=0, top_p=1.0,
                               repetition_penalty=1.0, banned_ids=None)
    new_tok = out[0][ids_t.shape[1]:]
    text = _tokenizer.decode(new_tok, skip_special_tokens=False)
    eos = '<｜end▁of▁sentence｜>'
    i = text.find(eos)
    if i >= 0:
        text = text[:i + len(eos)]
    try:
        parsed = parse_message_from_completion_text(text, 'thinking')
        return (parsed.get('reasoning_content') or ''), (parsed.get('content') or '').strip()
    except Exception:
        return '', text.replace(eos, '').replace('<think>', '').replace('</think>', '').strip()

def init_engine():
    global ENGINE_READY, _engine, _tokenizer, _wrapper, _agent
    if ENGINE_READY:
        return
    _engine = vs.V4FlashEngine()
    _tokenizer = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
    _wrapper = vs.V4ChatWrapper(_engine, _tokenizer)
    _agent = v4flash_agent.V4FlashAgent(engine=_engine, tokenizer=_tokenizer)
    ENGINE_READY = True

def reset_conversation():
    pass  # V4 无会话状态 (每次 generate 独立)

def generate_sync(msg, max_new=1048576):
    """阻塞生成完整文本 (供 stage_text 调用)"""
    if not ENGINE_READY:
        init_engine()
    return _wrapper.chat(msg, max_new_tokens=max_new)

def shutdown():
    """释放引擎内存, 让资源给后续阶段"""
    global ENGINE_READY, _engine, _tokenizer, _wrapper, _agent
    _engine = None; _tokenizer = None; _wrapper = None; _agent = None; ENGINE_READY = False
    gc.collect(); torch.cuda.empty_cache()

# ═══ HTTP 服务 (OpenAI 兼容, 供引擎管理器/CherryStudio 接入) ═══
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        if self.path.startswith('/v1/models'):
            body = json.dumps({'object':'list','data':[{'id':'deepseek-v4-flash','object':'model','owned_by':'local','permission':[]}]}).encode()
            self.send_response(200); self._cors(); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_response(404); self.end_headers(); return
        ln = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(ln).decode('utf-8', errors='replace'))
        messages = req.get('messages', [])
        # 提取最后一条 user 消息
        user_msg = ''
        for m in reversed(messages):
            if m.get('role') == 'user':
                user_msg = m.get('content', '')
                break
        max_new = 1048576  # 定格1M上下文 (忽略客户端 max_tokens, EOS自然停)
        try:
            if not ENGINE_READY:
                init_engine()
            # 读写头: 召回历史记忆, 拼接到当前消息 (与 35B 对齐)
            full_msg = user_msg
            recalled = recall_memory(user_msg)
            if recalled:
                ctx = '；'.join(f"历史({r['score']}): {r['user']} → {r['assistant'][:120]}" for r in recalled)
                full_msg = f"[历史相关记忆] {ctx}\n\n[当前问题] {user_msg}"
            if req.get('tools') or req.get('agent'):
                reasoning, text = '', chat_with_agent(full_msg, max_new_tokens=1048576)
            else:
                reasoning, text = chat_thinking(full_msg, max_new_tokens=max_new)
            if text:
                save_memory(user_msg, text)
            resp = {'id':'chatcmpl-v4flash','object':'chat.completion','created':int(time.time()),
                    'model':'deepseek-v4-flash',
                    'choices':[{'index':0,'message':{'role':'assistant','content':text,'reasoning_content':reasoning},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}
            body = json.dumps(resp, ensure_ascii=False).encode()
        except Exception as e:
            import traceback
            err = str(e) + '\n' + traceback.format_exc()
            try:
                with open(r'D:\ai\llamacpp\_v4flash_error.log', 'a', encoding='utf-8') as f:
                    f.write('[%s] %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), err))
            except Exception:
                pass
            body = json.dumps({'error': str(e)}).encode()
        self.send_response(200); self._cors(); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)

def serve(port=8123):
    init_engine()
    print(f'V4-Flash HTTP 服务 http://127.0.0.1:{port}/v1', flush=True)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
    srv.serve_forever()

if __name__ == '__main__':
    serve()

