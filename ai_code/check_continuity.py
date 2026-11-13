import json, os, struct

p = r'D:\ai\models\qwen3.5-397b-bf16'
idx = json.load(open(os.path.join(p, 'model.safetensors.index.json'), encoding='utf-8'))

targets = ['gate_up_proj', 'down_proj']
found = {}
for name, shard in idx['weight_map'].items():
    if 'layers.0.' in name and 'experts' in name and 'shared' not in name:
        for t in targets:
            if t in name:
                found[name] = shard

shard_info = {}
for name, shard in found.items():
    path = os.path.join(p, shard)
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    m = hdr[name]
    off_start, off_end = m['data_offsets']
    sz = off_end - off_start
    shard_info[name] = {
        'shard': shard, 'shape': m.get('shape'),
        'offset_start': off_start, 'offset_end': off_end,
        'size_MB': sz / 1048576, 'header_end': n + 8
    }
    print(f'{name}')
    print(f'  shard={shard} shape={m.get("shape")}')
    print(f'  data_offset=[{off_start}, {off_end}] size={shard_info[name]["size_MB"]:.0f}MB')
    per_expert = sz // 512
    print(f'  per_expert={per_expert} bytes = {per_expert/1048576:.1f}MB')
    print(f'  pread: fd.seek(header_end + expert_id * {sz//512})')
    print()

print('=== 连续性检查 ===')
shards_grouped = {}
for name, info in shard_info.items():
    shards_grouped.setdefault(info['shard'], []).append((name, info))
for sh, items in shards_grouped.items():
    items.sort(key=lambda x: x[1]['offset_start'])
    for i in range(len(items)-1):
        gap = items[i+1][1]['offset_start'] - items[i][1]['offset_end']
        tag = '紧邻' if gap == 0 else f'有间隙({gap}bytes)'
        print(f'  {sh}')
        print(f'    {items[i][0]} -> {items[i+1][0]} gap={gap} {tag}')

print()
print('=== 抽样验证: 多个层的gate_up_proj ===')
for layer_id in [0, 7, 15, 30, 45, 59]:
    for name, shard in idx['weight_map'].items():
        if f'layers.{layer_id}.' in name and 'gate_up_proj' in name and 'shared' not in name:
            path = os.path.join(p, shard)
            with open(path, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                hdr = json.loads(f.read(n))
            m = hdr[name]
            sz = m['data_offsets'][1] - m['data_offsets'][0]
            pe = sz / 512
            print(f'  layer {layer_id}: per_expert={pe/1048576:.2f}MB shape={m.get("shape")}')
            break
