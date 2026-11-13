"""V4-Flash 工具执行层 — 把模型输出的 DSML 工具调用映射到本机真实操作。

设计:
- 每个工具 = OpenAI function schema (喂给 encoding_dsv4.encode_messages) + 执行函数
- execute(name, arguments) 统一分发, 返回可 JSON 序列化的结果 (dict/list/str)
- 路径安全: 允许 C:/D: 全盘, 但黑名单保护系统关键目录
- 代码/命令执行: subprocess + 超时, 输出截断, 避免挂死/超长回喂
"""
import os, sys, json, time, re, subprocess, glob as _glob

# ──────────────────────────────────────────────
# 基础常量
# ──────────────────────────────────────────────
BASE_DIR = r'D:\ai'
MAIN_PY = r'D:\ai\python\python.exe'          # 主环境 (FLUX/LTX/TTS/通用依赖)
if not os.path.exists(MAIN_PY):
    MAIN_PY = sys.executable

# 系统关键目录黑名单 (写/删/执行保护, 读允许但写删禁止)
PROTECTED_PREFIXES = (
    r'C:\Windows', r'C:\Program Files', r'C:\Program Files (x86)',
    r'C:\ProgramData\Microsoft', r'C:\$Recycle.Bin', r'C:\System Volume Information',
)

# 文本文件扩展名 (read_file 只读这些, 避免读二进制)
TEXT_EXTS = {'.txt', '.md', '.py', '.json', '.yml', '.yaml', '.toml', '.log', '.cfg',
             '.ini', '.sh', '.bat', '.csv', '.tsv', '.js', '.ts', '.jsx', '.tsx',
             '.html', '.css', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.sql', '.xml'}

OUTPUT_LIMIT = 12000   # 工具结果回喂上限 (字符)
RUN_TIMEOUT = 180      # 代码/命令执行超时 (秒)


def _json(v):
    """安全序列化 (dict 里可能有非 JSON 类型则转 str)"""
    try:
        return json.loads(json.dumps(v, ensure_ascii=False, default=str))
    except Exception:
        return {'error': '结果不可序列化'}


def _truncate(s, limit=OUTPUT_LIMIT):
    s = '' if s is None else str(s)
    if len(s) > limit:
        return s[:limit] + f'\n...[截断, 共 {len(s)} 字符]'
    return s


def _resolve_path(path, base=BASE_DIR, for_write=False):
    """解析路径: 绝对路径直接用, 相对路径基于 base。返回 (abspath, error)。"""
    if path is None:
        return None, '路径为空'
    path = str(path).strip().strip('"').strip("'")
    if not path:
        path = '.'   # 空串 → 相对 base 根目录 (list_dir('') 表示 D:/ai)
    p = os.path.abspath(path if os.path.isabs(path) else os.path.join(base, path))
    # 规范化盘符大小写
    p_norm = p.lower()
    if for_write:
        for pre in PROTECTED_PREFIXES:
            if p_norm.startswith(pre.lower() + '\\') or p_norm == pre.lower():
                return None, f'拒绝操作: 路径 {p} 位于受保护系统目录'
    # 只允许存在的盘符 (C:/D:), 防 UNC/设备路径
    if len(p) < 2 or p[1] != ':':
        return None, f'不支持的路径格式: {p}'
    if p[0].upper() not in ('C', 'D'):
        return None, f'不支持的盘符: {p}'
    return p, None


# ──────────────────────────────────────────────
# 系统 / 联网
# ──────────────────────────────────────────────
def get_time():
    return {'time': time.strftime('%Y-%m-%d %H:%M:%S %A')}


def get_system_status():
    st = {'time': time.strftime('%Y-%m-%d %H:%M:%S %A')}
    try:
        import psutil
        st['cpu_percent'] = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        st['memory'] = {'total_gb': round(mem.total / 1024**3, 1),
                        'used_gb': round(mem.used / 1024**3, 1), 'percent': mem.percent}
        for d in ('C:\\', 'D:\\'):
            try:
                du = psutil.disk_usage(d)
                st[f'disk_{d[0].lower()}_free_gb'] = round(du.free / 1024**3, 1)
            except Exception:
                pass
    except Exception as e:
        st['psutil_error'] = str(e)[:120]
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        st['gpu'] = out or 'nvidia-smi 无输出'
    except Exception:
        st['gpu'] = 'nvidia-smi 不可用'
    return st


def web_search(query):
    if not query or not str(query).strip():
        return {'error': 'query 为空'}
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(str(query).strip(), max_results=8))
        return {'results': [{'title': r.get('title', ''), 'href': r.get('href', ''),
                             'body': r.get('body', '')[:300]} for r in results]}
    except Exception as e:
        return {'error': f'搜索失败: {str(e)[:200]}'}


def fetch_url(url):
    if not url or not str(url).strip():
        return {'error': 'url 为空'}
    url = str(url).strip()
    if not url.lower().startswith(('http://', 'https://')):
        return {'error': 'url 需以 http(s):// 开头'}
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode('utf-8', errors='replace')
        # 简单去标签
        text = re.sub(r'<script[\s\S]*?</script>|<style[\s\S]*?</style>', ' ', raw, flags=re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return {'url': url, 'content': text[:OUTPUT_LIMIT]}
    except Exception as e:
        return {'error': f'抓取失败: {str(e)[:200]}'}


# ──────────────────────────────────────────────
# 文件操作
# ──────────────────────────────────────────────
def list_dir(path=''):
    p, err = _resolve_path(path, base=BASE_DIR)
    if err:
        return {'error': err}
    if not os.path.isdir(p):
        return {'error': f'目录不存在: {p}'}
    items = []
    try:
        for name in os.listdir(p):
            full = os.path.join(p, name)
            try:
                if os.path.isdir(full):
                    items.append({'name': name, 'type': 'dir'})
                else:
                    items.append({'name': name, 'type': 'file', 'size': os.path.getsize(full)})
            except Exception:
                pass
    except Exception as e:
        return {'error': f'列目录失败: {str(e)[:200]}'}
    items.sort(key=lambda x: (x['type'] != 'dir', x['name'].lower()))
    return {'path': p, 'count': len(items), 'items': items[:120]}


def read_file(path):
    p, err = _resolve_path(path, base=BASE_DIR)
    if err:
        return {'error': err}
    if not os.path.isfile(p):
        return {'error': f'文件不存在: {p}'}
    ext = os.path.splitext(p)[1].lower()
    if ext not in TEXT_EXTS:
        return {'error': f'不支持文件类型 ({ext}), 只读文本类'}
    if os.path.getsize(p) > 500 * 1024:
        return {'error': f'文件太大 ({os.path.getsize(p)} 字节), 只读 ≤500KB'}
    try:
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            return {'path': p, 'content': f.read()}
    except Exception as e:
        return {'error': f'读取失败: {str(e)[:200]}'}


def write_file(path, content):
    p, err = _resolve_path(path, base=BASE_DIR, for_write=True)
    if err:
        return {'error': err}
    content = '' if content is None else str(content)
    try:
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        with open(p, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
        return {'ok': True, 'path': p, 'bytes': os.path.getsize(p), 'chars': len(content)}
    except Exception as e:
        return {'error': f'写入失败: {str(e)[:200]}'}


def delete_file(path):
    p, err = _resolve_path(path, base=BASE_DIR, for_write=True)
    if err:
        return {'error': err}
    if not os.path.exists(p):
        return {'error': f'路径不存在: {p}'}
    try:
        if os.path.isdir(p):
            import shutil
            shutil.rmtree(p)
            return {'ok': True, 'path': p, 'deleted': 'dir'}
        else:
            os.remove(p)
            return {'ok': True, 'path': p, 'deleted': 'file'}
    except Exception as e:
        return {'error': f'删除失败: {str(e)[:200]}'}


def search_files(pattern):
    if not pattern or not str(pattern).strip():
        return {'error': 'pattern 为空'}
    pattern = str(pattern).strip()
    try:
        hits = sorted(_glob.glob(pattern, recursive=True))[:100]
        return {'pattern': pattern, 'count': len(hits), 'files': hits}
    except Exception as e:
        return {'error': f'搜索失败: {str(e)[:200]}'}


def grep_content(pattern, path=''):
    if not pattern or not str(pattern).strip():
        return {'error': 'pattern 为空'}
    pat = str(pattern).strip()
    base, err = _resolve_path(path, base=BASE_DIR)
    if err:
        return {'error': err}
    base = base or BASE_DIR
    try:
        re.compile(pat)
    except Exception as e:
        return {'error': f'正则无效: {str(e)[:200]}'}
    matches = []
    files = []
    if os.path.isfile(base):
        files = [base]
    else:
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__', '.venv', 'venv')]
            for fn in fnames:
                if os.path.splitext(fn)[1].lower() in TEXT_EXTS:
                    files.append(os.path.join(root, fn))
            if len(files) > 5000:
                break
    for fp in files[:5000]:
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f, 1):
                    if re.search(pat, line):
                        matches.append({'file': fp, 'line': i, 'text': line.strip()[:200]})
                        if len(matches) >= 100:
                            break
        except Exception:
            continue
        if len(matches) >= 100:
            break
    return {'pattern': pat, 'path': base, 'matches': matches}


# ──────────────────────────────────────────────
# 代码 / 命令 / 环境
# ──────────────────────────────────────────────
def run_python(code):
    if not code or not str(code).strip():
        return {'error': 'code 为空'}
    code = str(code)
    try:
        out = subprocess.run([MAIN_PY, '-c', code], cwd=BASE_DIR,
                             capture_output=True, text=True, timeout=RUN_TIMEOUT,
                             encoding='utf-8', errors='replace')
        return {'ok': out.returncode == 0, 'returncode': out.returncode,
                'stdout': _truncate(out.stdout, 8000), 'stderr': _truncate(out.stderr, 4000)}
    except subprocess.TimeoutExpired:
        return {'error': f'代码执行超时 (>{RUN_TIMEOUT}s)'}
    except Exception as e:
        return {'error': f'执行失败: {str(e)[:200]}'}


def run_shell(command):
    if not command or not str(command).strip():
        return {'error': 'command 为空'}
    cmd = str(command)
    try:
        out = subprocess.run(['cmd', '/c', cmd], cwd=BASE_DIR,
                             capture_output=True, text=True, timeout=RUN_TIMEOUT,
                             encoding='utf-8', errors='replace')
        return {'ok': out.returncode == 0, 'returncode': out.returncode,
                'stdout': _truncate(out.stdout, 8000), 'stderr': _truncate(out.stderr, 4000)}
    except subprocess.TimeoutExpired:
        return {'error': f'命令执行超时 (>{RUN_TIMEOUT}s)'}
    except Exception as e:
        return {'error': f'执行失败: {str(e)[:200]}'}


def pip_install(packages):
    if not packages or not str(packages).strip():
        return {'error': 'packages 为空'}
    pkgs = str(packages).strip()
    try:
        out = subprocess.run([MAIN_PY, '-m', 'pip', 'install'] + pkgs.split(), cwd=BASE_DIR,
                             capture_output=True, text=True, timeout=600,
                             encoding='utf-8', errors='replace')
        tail = (out.stdout or '') + (out.stderr or '')
        return {'ok': out.returncode == 0, 'returncode': out.returncode,
                'output': _truncate(tail, 6000)}
    except subprocess.TimeoutExpired:
        return {'error': '安装超时 (>600s)'}
    except Exception as e:
        return {'error': f'安装失败: {str(e)[:200]}'}


# ──────────────────────────────────────────────
# 工具注册表 (OpenAI schema)
# ──────────────────────────────────────────────
TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "联网搜索互联网获取最新/实时信息（新闻、天气、价格、事件等）。返回标题+链接+摘要。", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "fetch_url", "description": "抓取指定网页 URL 的文本内容。", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "完整 http(s) URL"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "获取当前系统日期+星期+时间。", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_system_status", "description": "获取本机运行状态（CPU/内存/磁盘/显卡）。", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "list_dir", "description": "列出目录下的文件和子目录。path 可传绝对路径或相对 D:/ai 的路径。", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "目录路径（空串= D:/ai 根）"}}, "required": []}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取文本文件内容（.txt/.md/.py/.json 等）。", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "写文本文件（新建或覆盖），自动创建父目录。用于生成文件/写代码/存结果。", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "文件完整内容"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "delete_file", "description": "删除文件或目录（谨慎，不可恢复；系统关键目录会拒绝）。", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件或目录路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search_files", "description": "按 glob 模式搜索文件，如 'D:/ai/**/*.py' 或 '*.txt'。", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "glob 模式"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep_content", "description": "在文件内容里按正则搜索，返回匹配行（文件+行号+内容）。", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "正则表达式"}, "path": {"type": "string", "description": "文件或目录（默认 D:/ai）"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_python", "description": "运行一段 Python 代码，返回 stdout/stderr。用于写代码并执行。", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python 代码"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "run_shell", "description": "运行一条 shell 命令（cmd），返回输出。用于执行系统命令。", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "shell 命令"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "pip_install", "description": "用 pip 安装 Python 包（部署环境）。", "parameters": {"type": "object", "properties": {"packages": {"type": "string", "description": "包名，空格分隔，如 'requests numpy'"}}, "required": ["packages"]}}},
]

_EXECUTORS = {
    'web_search': web_search,
    'fetch_url': fetch_url,
    'get_time': get_time,
    'get_system_status': get_system_status,
    'list_dir': list_dir,
    'read_file': read_file,
    'write_file': write_file,
    'delete_file': delete_file,
    'search_files': search_files,
    'grep_content': grep_content,
    'run_python': run_python,
    'run_shell': run_shell,
    'pip_install': pip_install,
}


def execute(name, arguments):
    """执行单个工具。arguments 为 dict。返回可 JSON 序列化的结果。"""
    fn = _EXECUTORS.get(name)
    if fn is None:
        return {'error': f'未知工具: {name}'}
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except Exception:
            arguments = {}
    try:
        result = fn(**arguments) if isinstance(arguments, dict) else fn(arguments)
    except TypeError as e:
        return {'error': f'参数错误: {str(e)[:200]}'}
    except Exception as e:
        return {'error': f'工具异常: {str(e)[:200]}'}
    return _json(result)


if __name__ == '__main__':
    print(json.dumps(execute('get_time', {}), ensure_ascii=False))
    print(json.dumps(execute('list_dir', {'path': ''}), ensure_ascii=False, default=str)[:500])
