import winreg, os

out = r'D:\ai\_gpu_probe.txt'
lines = []

def dump(path, depth=0):
    try:
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
    except Exception:
        return
    try:
        i = 0
        while True:
            try:
                name, val, typ = winreg.EnumValue(k, i)
                lines.append('  ' * depth + name + ' = ' + str(val))
                i += 1
            except OSError:
                break
        j = 0
        while True:
            try:
                sub = winreg.EnumKey(k, j)
                lines.append('  ' * depth + '[' + sub + ']')
                dump(path + '\\' + sub, depth + 1)
                j += 1
            except OSError:
                break
    finally:
        winreg.CloseKey(k)

for base in [
    r'SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak',
    r'SOFTWARE\NVIDIA Corporation\Global\NVTweak',
    r'SOFTWARE\NVIDIA Corporation\Global',
]:
    lines.append('\n===== ' + base + ' =====')
    dump(base)

with open(out, 'a', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('注册表驱动设置已追加, 总新增行:', len(lines))
