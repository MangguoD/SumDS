简单的使用教程

conda activate sum_env
conda deactivate
python summarization_DS.py

v100不支持flash-attention
flash-attention环境需要提前在虚拟环境内编译（如果服务器总挂就晚上再试）

GLM:
source /root/DiabetesPDiagLLM/.diabetesPDiagLLMVenv/bin/activates

upload:
git status
git add .
git commit -m abcd
git push origin main

开发日志(author:MangguoD)：
2025/3/21:lv1
lv1:初始版本，基本完成了对数据的处理工作
lv1输出可用度在86.8%，速度为499条/3.39h

2025/3/22
lv2:1. 增加了约束力(也许是副作用？）
    2. 增加了输入提示模版，增强输出结构性，并且能够优化速度（不做retry的话平均加速4秒/条）
    3. 删除了眼底以及并发症的总结（因为大部分原数据都没有）
    4. 增加了输出后对规范性检测的内容，设置了宽容度，并且多次检测，提升输出质量和可用性（导致处理速度变长的主要原因）
        宽容度输出效果如下：
            [INVALID OUTPUT] 第 1 次尝试：结构不足 4 段
            第 2 次尝试成功：结构至少有 3 段
        也就是说，随着失败次数增加，对结构段落要求会逐渐降低
    5. 能够保存错误输出记录
    6. 增加了flash-attention支持的判断，自动选择
lv2undertest
