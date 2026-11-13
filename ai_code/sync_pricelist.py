# -*- coding: utf-8 -*-
"""[2026-10-04] 把主播价目表 xlsx 同步成网站数据 data/pricelist.json"""
import json, os, sys
try:
    import openpyxl
except Exception as e:
    print('NO_OPENPYXL ' + str(e)[:100]); sys.exit(1)
XLSX = r'C:\Users\Jinqi Wei\WPSDrive\1294570990\WPS云盘\主播初音泱未来代肝价目表（目前B站直播）.xlsx'
OUT = r'C:\Users\Jinqi Wei\AppData\Roaming\AutoClaw-official\accounts\95f2263891c467552a767999f335d2750fae3cd30d13820fa1fcc2896984fe50\projects\Default\smart-assistant\\data\\pricelist.json'
wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb['Sheet1']
rows = []
notice = []
for i, row in enumerate(ws.iter_rows(values_only=True), 1):
    vals = [('' if c is None else str(c).strip()) for c in row]
    if i <= 3:
        txt = ' '.join(v for v in vals if v)
        if txt:
            notice.append(txt)
        continue
    if i == 4:
        continue
    seq, game, item, price, dual, eta = (vals + [''] * 6)[:6]
    if not item:
        continue
    rows.append({'seq': seq, 'game': game, 'item': item, 'price': price, 'dual': dual, 'eta': eta})
data = {'updated': __import__('time').strftime('%Y-%m-%d %H:%M'), 'source': os.path.basename(XLSX),
        'notice': notice, 'rows': rows}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w', encoding='utf-8').write(json.dumps(data, ensure_ascii=False, indent=1))
print('已写出 %s：%d 条' % (OUT, len(rows)))
