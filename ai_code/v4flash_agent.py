"""V4-Flash Agent — 把引擎接上真实工具 (DSML 工具调用闭环)。

流程 (每轮):
  messages ──encode_messages──▶ prompt ──引擎生成──▶ DSML 文本
     ──parse_message_from_completion_text──▶ {content, tool_calls}
       有 tool_calls → 执行 v4flash_tools.execute → 回喂 role=tool → 下一轮
       无 tool_calls → 最终回答

依赖: v4flash_sched (引擎) + encoding_dsv4 (DSML 编解码) + v4flash_tools (工具执行)
"""
import os, sys, json, gc, time
import torch
sys.path.insert(0, r'D:\ai')
sys.path.insert(0, r'D:\ai\_v4flash_src')
import v4flash_sched as vs
import v4flash_tools as vtools
from encoding_dsv4 import encode_messages, parse_message_from_completion_text

DEVICE = vs.DEVICE
EOS = '<｜end▁of▁sentence｜>'

AGENT_SYSTEM = (
    '你是暄依嫣，卫锦旗的私人助手，运行在本地 DeepSeek-V4-Flash 模型上。'
    '你有能力通过工具真实操作这台电脑：读写/删除文件、写代码、运行代码、'
    '执行 shell 命令、安装环境、联网搜索与抓取网页、查看系统状态。'
    '当任务需要这些能力时，主动调用对应工具（一次可调多个），'
    '拿到工具结果后继续，直到完成用户目标再给出最终回答。'
    '最终回答用 Markdown，简洁准确，说明你做了什么、结果如何。'
)


class V4FlashAgent:
    def __init__(self, system_prompt=None, engine=None, tokenizer=None):
        # 可复用外部引擎/tokenizer (v4flash_server 场景, 避免双份引擎占双倍显存)
        self._owns_engine = engine is None
        if engine is not None:
            self.engine = engine
        else:
            self.engine = vs.V4FlashEngine()
        if tokenizer is not None:
            self.tok = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(str(vs.MODEL_DIR), local_files_only=True)
        self.system = system_prompt or AGENT_SYSTEM
        self._round_log = []

    # ── 单轮生成 ──
    def _generate(self, prompt, max_new_tokens=1048576, do_sample=False, temperature=0.6):
        ids = self.tok.encode(prompt, add_special_tokens=False)
        ids_t = torch.tensor([ids], device=DEVICE)
        with torch.inference_mode():
            out = self.engine.generate(ids_t, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                       temperature=temperature, top_k=0, top_p=1.0,
                                       repetition_penalty=1.0, banned_ids=None)
        new_tok = out[0][ids_t.shape[1]:]
        text = self.tok.decode(new_tok, skip_special_tokens=False)
        i = text.find(EOS)
        if i >= 0:
            text = text[:i + len(EOS)]
        return text

    # ── 主循环 ──
    def run(self, user_msg, tools=None, max_rounds=8, max_new_tokens=1048576, do_sample=False):
        tools = tools if tools is not None else vtools.TOOLS
        messages = [
            {'role': 'system', 'content': self.system, 'tools': tools},
            {'role': 'user', 'content': user_msg},
        ]
        self._round_log = []
        final_text = ''

        for rnd in range(max_rounds):
            prompt = encode_messages(messages, thinking_mode='chat')
            t0 = time.time()
            raw = self._generate(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
            try:
                parsed = parse_message_from_completion_text(raw, thinking_mode='chat')
            except Exception as e:
                # 格式错误: 直接返回原文 (去掉特殊 token), 不让 agent 卡死
                self._round_log.append({'round': rnd, 'tool_calls': [], 'error': str(e)[:200]})
                final_text = raw.replace(EOS, '').strip()
                break

            tool_calls = parsed.get('tool_calls') or []
            content = (parsed.get('content') or '').strip()
            self._round_log.append({'round': rnd, 'tool_calls': [tc['function']['name'] for tc in tool_calls],
                                    'content': content[:200], 'secs': round(time.time() - t0, 1)})

            if not tool_calls:
                final_text = content
                break

            # 回喂: assistant(含 tool_calls) + tool 结果
            messages.append({'role': 'assistant', 'content': content, 'tool_calls': tool_calls})
            for i, tc in enumerate(tool_calls):
                name = tc['function']['name']
                try:
                    args = json.loads(tc['function']['arguments'] or '{}')
                except Exception:
                    args = {}
                result = vtools.execute(name, args)
                messages.append({'role': 'tool', 'tool_call_id': str(i),
                                 'content': json.dumps(result, ensure_ascii=False)})
            final_text = content
        else:
            # 达到最大轮数: 返回最后一轮 content
            final_text = final_text or ''

        return {'text': final_text, 'rounds': len(self._round_log), 'log': self._round_log}

    def close(self):
        if self._owns_engine:
            self.engine = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    agent = V4FlashAgent()
    msg = sys.argv[1] if len(sys.argv) > 1 else '在 D:/ai 下写一个 hello.txt，内容写"你好，暄依嫣"，然后告诉我文件是否创建成功。'
    print('== 用户 ==', msg, flush=True)
    out = agent.run(msg, max_new_tokens=1048576, do_sample=False)
    print('== 最终回答 ==', flush=True)
    print(out['text'], flush=True)
    print('== 轮次日志 ==', flush=True)
    for r in out['log']:
        print(' ', r, flush=True)
    agent.close()
