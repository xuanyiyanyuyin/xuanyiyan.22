#!/bin/bash
# [2026-10-08] LingBot 二阶段开跑前检查（在远端实例上执行；只读，不删任何东西）
echo "=== 1) 磁盘与 inode ==="
df -h | head -20
echo "--- inode ---"
df -i | head -20
echo
echo "=== 2) 关键目录用量 ==="
for d in /workspace /root/rc /tmp; do [ -d "$d" ] && echo "[$d] $(du -sh $d 2>/dev/null | cut -f1)  文件数: $(find $d -xdev -type f 2>/dev/null | wc -l)"; done
echo
echo "=== 3) 是否已有历史产物需要先搬走 ==="
ls -lh /root/rc/out 2>/dev/null | head -10 || echo "(无 /root/rc/out)"
echo
echo "=== 4) 大文件 top10（>1G）==="
find / -xdev -type f -size +1G 2>/dev/null | head -10
echo
echo "=== 5) GPU 与内存 ==="
rocm-smi 2>/dev/null | head -20 || nvidia-smi 2>/dev/null | head -20 || echo "(无 GPU 工具)"
free -g
echo
echo "=== 6) 结论提示 ==="
echo "· inode 使用率 > 60% 或 临时盘 > 100GB ⇒ 先清/搬，再开跑"
echo "· 开跑命令与存点间隔见 D:\\ai\\LingBot_二阶段计划_20260929.md"