#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[2026-10-02] QQ 官方机器人 → OneBot v11 网关（喵朵官方通道）

作用：把 QQ 开放平台的官方机器人协议翻译成 OneBot v11，让 qq_bot.py 一行不改就能跑官方通道。
  · 官方 → 我们：WSS 收事件 → 转 OneBot 事件 → POST http://127.0.0.1:8899
  · 我们 → 官方：本进程监听 3000（冒充 NapCat）提供 /send_group_msg /send_private_msg
                /get_image /get_login_info /get_status → 翻成官方 v2 接口

官方协议要点（2026-10-02 核对官方文档）：
  · 鉴权 POST https://bots.qq.com/app/getAppAccessToken {appId, clientSecret} → access_token，7200s
  · 网关 GET https://api.sgroup.qq.com/gateway（Authorization: QQBot {token}）
  · op10 HELLO(heartbeat_interval 45000) → op2 Identify{token, intents, shard:[0,1]}
  · op1 心跳；op6 Resume{token,session_id,seq}；op0 事件（t=事件类型）
  · intents GROUP_AND_C2C_EVENT(1<<25) → C2C_MESSAGE_CREATE(单聊) / GROUP_AT_MESSAGE_CREATE(群@)
  · 发消息 POST /v2/users/{openid}/messages 或 /v2/groups/{group_openid}/messages
    只能被动回复：必须带 msg_id（群窗口 5 分钟、单聊 60 分钟，每条最多 5 次，msg_seq 递增）
  · 富媒体只能传 url（file_data 官方暂未支持）

ID 映射（官方只给 openid）：主人 openid → 伪装成 1503163696；群里 @ 事件补 at 段 qq=SELF_QQ
日志：D:\ai\llamacpp\qqgw.log
"""
import asyncio
import json
import os
import sys
import time
import traceback
import urllib.parse

import secrets

import aiohttp
from aiohttp import web

CFG_PATH = r'D:\ai\qq_official.json'
LOG_PATH = r'D:\ai\llamacpp\qqgw.log'
API_BASE = 'https://api.sgroup.qq.com'
TOKEN_URL = 'https://bots.qq.com/app/getAppAccessToken'
INTENTS = 1 << 25
QQBOT_URL = 'http://127.0.0.1:8899'
SELF_QQ = 3900148698
BOT_PLATFORM_ID = '5504400249196725410'   # 官方平台里本机器人自己的 id（用于判断 mentions 里是否 @ 了它）
AT_USE_TAG = (os.environ.get('AT_USE_TAG', '0') == '1')   # [2026-10-27] 默认关官方<@openid>标签(群里会显示原文)   # [2026-10-08] @ 用官方标签 <@openid>；设 0 退回纯文本
OWNER_QQ = '1503163696'
MEDIA_DIR = r'D:\ai\llamacpp\cache\official'
CTX_PATH = r'D:\ai\llamacpp\qqgw_ctx.json'   # 会话上下文(msg_id)持久化
MEDIA_PORT = 8896                       # 本地媒体静态服务端口
MEDIA_ROOT = r'D:\ai\output'           # 对外暴露的生成物目录（只此一个）
CF_BIN = r'D:\ai\bin\cloudflared.exe'  # cloudflared（快速隧道，免账号）
GROUP_WINDOW = 300
C2C_WINDOW = 3600
MAX_REPLY_PER_MSG = 5


def log(msg):
    line = '[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def load_cfg():
    with open(CFG_PATH, encoding='utf-8') as f:
        return json.load(f)


def save_cfg(cfg):
    try:
        with open(CFG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log('保存配置失败: %s' % str(e)[:120])



def parse_ts(v):
    """官方事件的 timestamp 是 ISO8601 字符串（如 2026-10-02T16:14:53+08:00），不是 epoch 数字。
    2026-10-02 踩坑：直接 int() 会抛 ValueError，把事件处理整个打断（现象=喵朵没输出）。"""
    try:
        return int(float(v))
    except Exception:
        pass
    try:
        import datetime
        return int(datetime.datetime.fromisoformat(str(v).replace('Z', '+00:00')).timestamp())
    except Exception:
        return int(time.time())

def _kill_old_tunnels():
    """[2026-10-02] 启动隧道前先杀掉遗留的 cloudflared
    （原网关每次重启都会新起一条隧道且不回收旧进程，实测泄漏到 12 个公网入口）"""
    try:
        import subprocess as _sp
        r = _sp.run(['taskkill', '/F', '/IM', 'cloudflared.exe'], capture_output=True, timeout=20)
        out = (r.stdout or b'').decode('gbk', 'replace')
        if 'SUCCESS' in out.upper() or chr(25104) + chr(21151) in out:
            log('已清理遗留 cloudflared 隧道进程')
    except Exception as e:
        log('清理旧隧道失败(忽略): %s' % str(e)[:80])


class Gateway:
    def __init__(self):
        self.cfg = load_cfg()
        self.token = None
        self.token_exp = 0
        self.session = None
        self.ws = None
        self.seq = 0
        self.session_id = None
        self.heartbeat_iv = 45
        self.owner_map = self.cfg.get('owner_map') or {}
        self.msg_ctx = {}
        try:
            self.msg_ctx = json.load(open(CTX_PATH, encoding='utf-8')) or {}
        except Exception:
            self.msg_ctx = {}
        self.media_seen = {}
        # openid/显示ID -> 群昵称（官方没有真 @，只能把 @ 渲染成人名）
        self.names = self.cfg.get('names') or {}
        # 显示ID -> openid（单聊：qq_bot 传回来的 user_id 是映射后的号）
        self.uid2open = {v: k for k, v in (self.cfg.get('owner_map') or {}).items()}
        self.tunnel_url = ''
        self.tunnel_proc = None
        self.media_token = self.cfg.get('media_token') or secrets.token_urlsafe(12)
        if self.cfg.get('media_token') != self.media_token:
            self.cfg['media_token'] = self.media_token
            save_cfg(self.cfg)
        os.makedirs(MEDIA_DIR, exist_ok=True)

    async def ensure_token(self):
        if self.token and time.time() < self.token_exp - 120:
            return self.token
        body = {'appId': str(self.cfg['app_id']), 'clientSecret': self.cfg['app_secret']}
        async with self.session.post(TOKEN_URL, json=body, timeout=aiohttp.ClientTimeout(total=20)) as r:
            d = await r.json(content_type=None)
        tk = d.get('access_token')
        if not tk:
            raise RuntimeError('取 token 失败: %s' % str(d)[:200])
        self.token = tk
        self.token_exp = time.time() + int(d.get('expires_in', 7200))
        log('access_token 已刷新（%ss）' % d.get('expires_in'))
        return tk

    async def api(self, method, path, **kw):
        tk = await self.ensure_token()
        headers = {'Authorization': 'QQBot ' + tk, 'Content-Type': 'application/json'}
        async with self.session.request(method, API_BASE + path, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=20), **kw) as r:
            txt = await r.text()
            try:
                return r.status, json.loads(txt)
            except Exception:
                return r.status, {'raw': txt[:300]}

    async def ws_loop(self):
        while True:
            try:
                await self.ws_once()
            except Exception as e:
                log('WSS 异常: %s: %s' % (type(e).__name__, str(e)[:160]))
            await asyncio.sleep(5)

    async def ws_once(self):
        tk = await self.ensure_token()
        st, d = await self.api('GET', '/gateway')
        url = d.get('url') if isinstance(d, dict) else None
        if not url:
            raise RuntimeError('取网关失败 HTTP %s %s' % (st, str(d)[:160]))
        log('连接网关 %s' % url)
        async with self.session.ws_connect(url, heartbeat=None, timeout=30) as ws:
            self.ws = ws
            hb_task = None
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                try:
                    p = json.loads(msg.data)
                except Exception:
                    continue
                op = p.get('op')
                if p.get('s'):
                    self.seq = p['s']
                if op == 10:
                    self.heartbeat_iv = int((p.get('d') or {}).get('heartbeat_interval', 45000)) / 1000.0
                    if self.session_id:
                        await ws.send_json({'op': 6, 'd': {'token': 'QQBot ' + tk,
                                                           'session_id': self.session_id, 'seq': self.seq}})
                        log('发送 Resume（seq=%s）' % self.seq)
                    else:
                        await ws.send_json({'op': 2, 'd': {'token': 'QQBot ' + tk,
                                                           'intents': INTENTS, 'shard': [0, 1]}})
                        log('发送 Identify（intents=%d）' % INTENTS)
                    if hb_task is None:
                        hb_task = asyncio.create_task(self.heartbeat(ws))
                elif op == 0:
                    t = p.get('t')
                    if t == 'READY':
                        dd = p.get('d') or {}
                        self.session_id = dd.get('session_id')
                        if (dd.get('user') or {}).get('id'):
                            self.bot_id = str(dd['user']['id'])
                        u = dd.get('user') or {}
                        log('READY: 机器人 %s(%s) 已上线' % (u.get('username'), u.get('id')))
                    else:
                        try:
                            await self.on_event(t, p.get('d') or {})
                        except Exception as _e:
                            log('事件处理异常 %s: %s' % (t, traceback.format_exc()[-400:]))
                elif op == 7:
                    log('服务端要求重连')
                    break
                elif op == 9:
                    log('Invalid Session → 重新 Identify')
                    self.session_id = None
                    break
            if hb_task:
                hb_task.cancel()
        self.ws = None

    async def heartbeat(self, ws):
        while True:
            await asyncio.sleep(self.heartbeat_iv)
            try:
                await ws.send_json({'op': 1, 'd': self.seq})
            except Exception:
                return

    def display_id(self, openid):
        if openid in self.owner_map:
            self.uid2open[self.owner_map[openid]] = openid
            return self.owner_map[openid]
        # [2026-10-02 主人令·写死] 原「首个私聊者自动成为主人」兜底已删除：
        #   认主唯一来源 = qq_official.json 的 owner_map；不在名单里的既不是主人，也绝不自动认主。
        log('非主人 openid(%s…)：owner_map 未收录（群/私聊事件都可能触发本行）' % openid[:10])
        return openid

    def save_ctx(self):
        """把每个会话最新的 msg_id 落盘：网关重启后仍能在窗口内被动回复"""
        try:
            keep = {k: v for k, v in self.msg_ctx.items() if time.time() - v.get('ts', 0) < 3600}
            self.msg_ctx = keep
            with open(CTX_PATH, 'w', encoding='utf-8') as f:
                json.dump(keep, f, ensure_ascii=False)
        except Exception as e:
            log('保存会话上下文失败: %s' % str(e)[:120])

    async def on_event(self, t, d):
        # [2026-10-02] 支持三类：群@、单聊、以及"群全量消息"（平台开启
        #   「机器人可获取的群聊消息范围=获取群内全部消息」后才会推，事件名通常是
        #   GROUP_MESSAGE_CREATE / GROUP_ADD_ROBOT 之类的 *MESSAGE_CREATE）——
        #   全量消息【不补 at 段】，这样 qq_bot 走它的关键词触发(GROUP_TRIGGERS)决定回不回。
        is_at = (t == 'GROUP_AT_MESSAGE_CREATE')
        is_full = (t != 'GROUP_AT_MESSAGE_CREATE' and t != 'C2C_MESSAGE_CREATE'
                   and t.endswith('MESSAGE_CREATE') and bool(d.get('group_openid')))
        if not (is_at or is_full or t == 'C2C_MESSAGE_CREATE'):
            log('忽略事件 %s' % t)
            return
        is_group = (is_at or is_full)
        content = (d.get('content') or '').strip()
        # [2026-10-27 主人令] 官方全量消息里会带 <@openid> 提及标记：不能原样给模型/日志，
        #   否则她复述出来就是一串 openid。能查到昵称就转 @昵称，查不到就删掉。
        try:
            import re as _re_at
            content = _re_at.sub(r'<@!?([0-9A-Fa-f]{16,})>',
                                 lambda _m: ('@' + self.names.get(_m.group(1), '某人')), content)
        except Exception:
            pass
        author = d.get('author') or {}
        openid = author.get('member_openid') or author.get('user_openid') or ''
        uid = self.display_id(openid)
        # [2026-10-02 修] 官方 User 对象的昵称字段是 username（nickname 恒为空），
        #   之前读错 ⇒ 昵称表一直是空的 ⇒ @ 只回退成 openid。
        _nick = (author.get('username') or author.get('nickname') or '').strip()
        if not _nick:
            log('作者字段（无昵称，供核对）: %s' % json.dumps(author, ensure_ascii=False)[:200])
        if _nick:
            self.names[str(openid)] = _nick
            self.names[str(uid)] = _nick          # 主人号(1503163696)也映射到同一昵称
            if self.cfg.get('names') != self.names:
                self.cfg['names'] = self.names
                save_cfg(self.cfg)
        conv = (d.get('group_openid') if is_group else openid) or ''
        self.msg_ctx[conv] = {'msg_id': d.get('id'), 'ts': time.time(), 'seq': 0,
                             'kind': 'group' if is_group else 'c2c'}
        segs = []
        if is_at:
            segs.append({'type': 'at', 'data': {'qq': str(SELF_QQ)}})   # 群@ → 补 at 段让 at_me 成立
        elif is_full:
            log('群全量消息事件（%s）→ 交给关键词触发判定' % t)
        if content:
            segs.append({'type': 'text', 'data': {'text': content}})
        for att in (d.get('attachments') or []):
            u = att.get('url') or ''
            ct = (att.get('content_type') or '').lower()
            if not u:
                continue
            if any(k in ct for k in ('image', 'png', 'jpeg', 'jpg', 'webp')):
                local = await self.fetch_media(u)
                segs.append({'type': 'image', 'data': {'file': local or u, 'url': u}})
        # [2026-10-02] 官方全量群消息里的 mentions（谁被 @ 了）→ 转成 OneBot at 段，
        #   这样 qq_bot 的 at_me 判定对"@ 她"同样成立（原来只认显式 @ 事件）。
        if d.get('mentions'):
            log('mentions: %s' % [str(m.get('id') or m.get('user_openid') or '')[:12] for m in d['mentions']])
        for _mn in (d.get('mentions') or []):
            _mid = str(_mn.get('id') or _mn.get('user_openid') or _mn.get('member_openid') or '')
            if not _mid:
                continue
            # [2026-10-02 修] 群里 @ 她给的的是「群内 openid」(如 C318D7F…)，不是平台 id
            #   ⇒ 只要命中 self_openids 就当作 @ 了机器人（at_me 才会成立）
            _self_ids = set(str(x) for x in (self.cfg.get('self_openids') or []))
            _self_ids |= {str(SELF_QQ), str(BOT_PLATFORM_ID), str(getattr(self, 'bot_id', ''))}
            _qq = str(SELF_QQ) if _mid in _self_ids else _mid
            segs.append({'type': 'at', 'data': {'qq': _qq}})
        ev = {'post_type': 'message',
              'message_type': 'group' if is_group else 'private',
              'self_id': SELF_QQ,
              'user_id': uid,
              'time': parse_ts(d.get('timestamp')),
              'message_id': d.get('id'),
              'message': segs,
              'raw_message': content,
              'sender': {'user_id': uid, 'nickname': _nick or str(uid),
                         'card': _nick or '', 'role': 'member', 'openid': openid,
                         'bot': bool(author.get('bot'))}}
        if is_group:
            ev['group_id'] = d.get('group_openid')
            ev['group_openid'] = d.get('group_openid')
        _kind = '群@' if is_at else ('群全量' if is_full else '单聊')
        log('收到%s消息 [%s] %s' % (_kind, str(uid)[:14], content[:300]))   # [2026-11-12] 60→300: 主人令,提示词要能进监控
        self.save_ctx()
        # [2026-10-02] 失败自动重试：qq_bot 重启那几秒连不上，原来直接把消息丢了
        for _try in range(4):
            try:
                async with self.session.post(QQBOT_URL + '/', json=ev,
                                             timeout=aiohttp.ClientTimeout(total=20)) as r:
                    await r.text()
                break
            except Exception as e:
                if _try < 3:
                    log('转发 qq_bot 失败（第%d次），%.1fs 后重试: %s' % (_try + 1, 1.5, str(e)[:90]))
                    await asyncio.sleep(1.5)
                else:
                    log('转发给 qq_bot 失败（已重试 4 次，本条丢弃）: %s' % str(e)[:140])

    async def fetch_media(self, url):
        if url in self.media_seen and os.path.exists(self.media_seen[url]):
            return self.media_seen[url]
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                data = await r.read()
            name = 'att_%d_%s' % (int(time.time()),
                                  os.path.basename(urllib.parse.urlparse(url).path)[-24:] or 'img')
            if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                name += '.png'
            p = os.path.join(MEDIA_DIR, name)
            with open(p, 'wb') as f:
                f.write(data)
            self.media_seen[url] = p
            log('附件已落盘 %s (%.1f KB)' % (os.path.basename(p), len(data) / 1024))
            return p
        except Exception as e:
            log('附件下载失败: %s' % str(e)[:140])
            return ''

    def segs_to_text(self, message):
        if isinstance(message, str):
            return message
        out = []
        for s in (message or []):
            typ = s.get('type')
            dat = s.get('data') or {}
            if typ == 'text':
                out.append(dat.get('text', ''))
            elif typ == 'at':
                # [2026-10-08] 原来只拼 '@昵称' ⇒ 是【纯文本假 @】，官方不会渲染成提及
                # 改为官方提及标签 <@openid>；拿不到有效 openid 时退回 ⓐ昵称 文本
                _k = str(dat.get('qq', ''))
                _open = str(self.uid2open.get(_k, _k) or '')
                _bad = (not _open) or _open in ('0', 'all', 'ALL') or _open == str(SELF_QQ)
                if AT_USE_TAG and not _bad:
                    out.append('<@%s> ' % _open)   # [2026-10-13] 官方标签后补空格，否则不被解析成 @ 而显示原文
                else:
                    _nm = self.names.get(_k, '')
                    if not _nm:
                        _nm = '群友' if (len(_k) >= 16 and all(c in '0123456789abcdefABCDEF' for c in _k)) else _k
                    out.append('@' + _nm)
            elif typ == 'face':
                out.append('[表情]')
        return ''.join(out)

    def text_only(self, message):
        """只取 text 段（用于 markdown@ 正文，避免把 at 段再渲染成 '@昵称' 重复一次）"""
        out = []
        for s in (message or []):
            _t = s.get('type')
            if _t == 'text':
                out.append((s.get('data') or {}).get('text', ''))
            elif _t == 'face':
                out.append('[表情]')
        return ''.join(out)
    def group_markdown(self, message, text):
        """群聊真 @：官方要求 markdown + <qqbot-at-user id="member_openid" />（2026-10-27 查文档）"""
        parts = []
        has_at = False
        for s in (message or []):
            if s.get('type') != 'at':
                continue
            _k = str((s.get('data') or {}).get('qq', ''))
            _open = str(self.uid2open.get(_k, _k) or '')
            if not _open or _open in ('0', 'all', 'ALL') or _open == str(SELF_QQ):
                continue
            parts.append('<qqbot-at-user id="%s" /> ' % _open)
            has_at = True
        if not has_at:
            return ''
        parts.append(text or '')
        return ''.join(parts)[:4000]
    def media_segs(self, message):
        if isinstance(message, str):
            return []
        return [s for s in (message or []) if s.get('type') in ('image', 'video', 'record', 'file')]

    async def official_send(self, conv, kind, message, group_openid=None, user_openid=None):
        ctx = self.msg_ctx.get(conv) or {}
        msg_id = ctx.get('msg_id')
        window = GROUP_WINDOW if kind == 'group' else C2C_WINDOW
        if not msg_id or (time.time() - ctx.get('ts', 0)) > window:
            log('被动窗口已过期（%s），官方拒绝发送；需用户再发一条消息刷新' % kind)
            return {'status': 'failed', 'retcode': 100, 'wording': 'passive window expired'}
        ctx['seq'] = int(ctx.get('seq', 0)) + 1
        if ctx['seq'] > MAX_REPLY_PER_MSG:
            log('本条消息回复次数已达上限 %d' % MAX_REPLY_PER_MSG)
            return {'status': 'failed', 'retcode': 101, 'wording': 'reply limit reached'}
        path = ('/v2/groups/%s/messages' % group_openid) if kind == 'group' \
            else ('/v2/users/%s/messages' % user_openid)
        text = self.segs_to_text(message)
        medias = self.media_segs(message)
        results = []
        if text.strip() or medias == []:
            _md = self.group_markdown(message, self.text_only(message)) if kind == 'group' else ''
            _sent = False
            if _md:
                _body = {'markdown': {'content': _md}, 'msg_type': 2, 'msg_id': msg_id, 'msg_seq': ctx['seq']}
                st, d = await self.api('POST', path, json=_body)
                if st == 200 and not (isinstance(d, dict) and (d.get('code') or d.get('err_code'))):
                    results.append((st, d))
                    log('发 markdown@ %d 字 → HTTP %s' % (len(_md), st))
                    _sent = True
                else:
                    log('markdown@ 发送失败(HTTP %s %s)，回退文本' % (st, str(d)[:160]))
            if not _sent and text.strip():
                body = {'content': text[:4000], 'msg_type': 0, 'msg_id': msg_id, 'msg_seq': ctx['seq']}
                st, d = await self.api('POST', path, json=body)
                results.append((st, d))
                log('发文本 %d 字 → HTTP %s %s' % (len(text), st, str(d)[:130]))
        for m in medias:
            local = (m.get('data', {}).get('file', '') or '').replace('file:///', '').replace('/', os.sep)
            if not os.path.exists(local):
                log('媒体文件不存在，跳过: %s' % local[:120])
                continue
            # [2026-11-05 修] 图文同发时媒体与文本共用同一 msg_seq 会被官方 40054005 去重 → 媒体单独递增
            ctx['seq'] = int(ctx.get('seq', 0)) + 1
            if ctx['seq'] > MAX_REPLY_PER_MSG:
                log('本条消息回复次数已达上限 %d(媒体)' % MAX_REPLY_PER_MSG)
                break
            r = await self.send_media(kind, msg_id, ctx['seq'], group_openid or user_openid, local)
            if r:
                results.append((200 if r.get('status') == 'ok' else 0, r))
            else:
                results.append((0, {'error': 'no public url'}))
        ok = any(st == 200 for st, _ in results) if results else False
        if ok:
            return {'status': 'ok', 'retcode': 0, 'data': {'message_id': ctx['seq']}}
        return {'status': 'failed', 'retcode': 102,
                'wording': str(results[-1][1])[:200] if results else 'empty'}

    # ── 公网 URL：本地静态服务 + cloudflared 快速隧道 ──
    def media_routes(self):
        async def serve(req):
            tok = req.match_info.get('token')
            if tok != self.media_token:
                return web.Response(status=404, text='not found')
            rel = req.match_info.get('tail') or ''
            rel = urllib.parse.unquote(rel).replace('/', os.sep)
            full = os.path.abspath(os.path.join(MEDIA_ROOT, rel))
            if not full.startswith(os.path.abspath(MEDIA_ROOT)) or not os.path.isfile(full):
                return web.Response(status=404, text='not found')
            return web.FileResponse(full)
        return serve

    def public_url_for(self, local):
        """本地文件 → 公网 URL（走隧道）；不在 MEDIA_ROOT 里的返回空"""
        try:
            full = os.path.abspath(local)
            root = os.path.abspath(MEDIA_ROOT)
            if not full.startswith(root) or not os.path.isfile(full):
                return ''
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            rel = urllib.parse.quote(rel)
            if not self.tunnel_url:
                return ''
            return '%s/m/%s/%s' % (self.tunnel_url, self.media_token, rel)
        except Exception:
            return ''

    async def start_tunnel(self):
        # [2026-10-12 主人令] 固定媒体域名优先：有 media_base 就不再启 cloudflared 快速隧道
        _fixed = (self.cfg.get('media_base') or os.environ.get('MEDIA_BASE_URL') or '').strip()
        if _fixed:
            self.tunnel_url = _fixed.rstrip('/')
            log('使用固定媒体域名: %s（跳过 cloudflared）' % self.tunnel_url)
            return
        """启动 cloudflared 快速隧道并把公网地址抓出来；断了自动重来"""
        _kill_old_tunnels()      # [2026-10-02] 先清旧隧道，防止公网入口泄漏
        if not os.path.exists(CF_BIN):
            log('未找到 cloudflared（%s），富媒体发送将不可用' % CF_BIN)
            return
        while True:
            try:
                proc = await asyncio.create_subprocess_exec(
                    CF_BIN, 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:%d' % MEDIA_PORT,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                self.tunnel_proc = proc
                log('cloudflared 已启动，等待公网地址…')
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    s = line.decode('utf-8', 'replace').strip()
                    if 'trycloudflare.com' in s and 'https://' in s:
                        import re as _re
                        m = _re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', s)
                        if m and m.group(0) != self.tunnel_url:
                            self.tunnel_url = m.group(0)
                            log('公网地址就绪: %s（富媒体走这里）' % self.tunnel_url)
                    elif 'ERR' in s or 'error' in s.lower():
                        log('cloudflared: %s' % s[:160])
                log('cloudflared 退出，5 秒后重启')
            except Exception as e:
                log('cloudflared 异常: %s: %s' % (type(e).__name__, str(e)[:140]))
            self.tunnel_url = ''
            await asyncio.sleep(5)

    def _ftype(self, path):
        e = os.path.splitext(path)[1].lower()
        if e in ('.png', '.jpg', '.jpeg', '.webp', '.gif'):
            return 1
        if e in ('.mp4', '.mov'):
            return 2
        if e in ('.silk', '.amr'):
            return 3
        if e in ('.mp3', '.wav', '.ogg'):
            return 3          # 官方要 silk；先按语音试，失败会记日志
        return 1

    async def send_media(self, kind, msg_id, seq, openid_or_group, local):
        """富媒体：先上传拿 file_info，再发 msg_type=7（被动回复）"""
        url = self.public_url_for(local)
        if not url:
            log('媒体无法生成公网 URL（不在 %s 内或隧道未就绪）: %s' % (MEDIA_ROOT, local[:100]))
            return None
        up_path = ('/v2/groups/%s/files' % openid_or_group) if kind == 'group' \
            else ('/v2/users/%s/files' % openid_or_group)
        body = {'file_type': self._ftype(local), 'url': url, 'srv_send_msg': False}
        st, d = await self.api('POST', up_path, json=body)
        fi = (d or {}).get('file_info')
        log('上传媒体 HTTP %s file_type=%s → file_info=%s url=%s' % (st, body['file_type'], bool(fi), url[:90]))
        if not fi:
            return {'status': 'failed', 'retcode': 103, 'wording': str(d)[:200]}
        send_path = ('/v2/groups/%s/messages' % openid_or_group) if kind == 'group' \
            else ('/v2/users/%s/messages' % openid_or_group)
        st2, d2 = await self.api('POST', send_path, json={'msg_type': 7, 'media': {'file_info': fi},
                                                          'msg_id': msg_id, 'msg_seq': seq})
        log('发媒体 → HTTP %s seq=%s %s' % (st2, seq, str(d2)[:160]))
        # [2026-11-05 修] 官方 40054005 = msg_seq 撞车 → seq 递增重试(最多 3 次)
        _tries = 0
        while st2 == 400 and isinstance(d2, dict) and d2.get('err_code') == 40054005 and _tries < 3:
            _tries += 1
            seq = int(seq) + 1
            st2, d2 = await self.api('POST', send_path, json={'msg_type': 7, 'media': {'file_info': fi},
                                                              'msg_id': msg_id, 'msg_seq': seq})
            log('发媒体重试 seq=%s → HTTP %s %s' % (seq, st2, str(d2)[:120]))
        if st2 == 200:
            return {'status': 'ok', 'retcode': 0, 'data': d2}
        return {'status': 'failed', 'retcode': 104, 'wording': str(d2)[:200]}

    async def h_send_group(self, req):
        b = await req.json()
        gid = str(b.get('group_id', ''))
        return web.json_response(await self.official_send(gid, 'group', b.get('message', ''), group_openid=gid))

    async def h_send_private(self, req):
        b = await req.json()
        uid = str(b.get('user_id', ''))
        openid = self.uid2open.get(uid, uid)      # 显示ID → 真实 openid（会话键一致才能拿到 msg_id）
        if openid != uid:
            log('单聊发送: user_id=%s → openid=%s...' % (uid, openid[:10]))
        return web.json_response(await self.official_send(openid, 'c2c', b.get('message', ''),
                                                          user_openid=openid))

    async def h_get_login_info(self, req):
        return web.json_response({'status': 'ok', 'retcode': 0,
                                  'data': {'user_id': SELF_QQ, 'nickname': '喵朵'}})

    async def h_get_status(self, req):
        online = self.ws is not None
        return web.json_response({'status': 'ok', 'retcode': 0,
                                  'data': {'online': online, 'good': online}})

    async def h_get_image(self, req):
        b = await req.json()
        ref = str(b.get('file', ''))
        p = ref.replace('file:///', '')
        if os.path.exists(p):
            return web.json_response({'status': 'ok', 'retcode': 0, 'data': {'file': p}})
        if ref.startswith('http'):
            local = await self.fetch_media(ref)
            if local:
                return web.json_response({'status': 'ok', 'retcode': 0, 'data': {'file': local}})
        return web.json_response({'status': 'failed', 'retcode': 1404, 'wording': 'image not found'})

    async def h_fallback(self, req):
        try:
            b = await req.json()
        except Exception:
            b = {}
        log('未实现的 OneBot 动作: %s %s' % (req.path, str(b)[:100]))
        return web.json_response({'status': 'failed', 'retcode': 1404,
                                  'wording': 'gateway: not implemented'})

    def build_app(self):
        app = web.Application()
        app.router.add_post('/send_group_msg', self.h_send_group)
        app.router.add_post('/send_private_msg', self.h_send_private)
        app.router.add_post('/send_msg', self.h_send_group)
        app.router.add_post('/get_image', self.h_get_image)
        app.router.add_get('/get_login_info', self.h_get_login_info)
        app.router.add_get('/get_status', self.h_get_status)
        app.router.add_get('/health', lambda r: web.json_response({'ok': True, 'ws': self.ws is not None,
                                                                   'tunnel': self.tunnel_url}))
        app.router.add_get('/m/{token}/{tail:.*}', self.media_routes())
        app.router.add_route('*', '/{tail:.*}', self.h_fallback)
        return app


async def _media_app(gw):
    app = web.Application()
    app.router.add_get('/m/{token}/{tail:.*}', gw.media_routes())
    return app


async def main():
    gw = Gateway()
    gw.session = aiohttp.ClientSession()
    log('QQ 官方网关启动 appid=%s 监听 3000 → 事件转发 %s' % (gw.cfg['app_id'], QQBOT_URL))
    runner = web.AppRunner(gw.build_app())
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 3000)
    await site.start()
    # 媒体静态服务（只暴露 MEDIA_ROOT，带随机 token）
    mrunner = web.AppRunner(await _media_app(gw))
    await mrunner.setup()
    await web.TCPSite(mrunner, '127.0.0.1', MEDIA_PORT).start()
    log('媒体静态服务 http://127.0.0.1:%d/m/<token>/（只服务 %s）' % (MEDIA_PORT, MEDIA_ROOT))
    asyncio.create_task(gw.start_tunnel())
    log('OneBot API 已监听 http://127.0.0.1:3000（/health 查状态）')
    await gw.ws_loop()


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        log('网关崩溃: %s' % traceback.format_exc()[-800:])
