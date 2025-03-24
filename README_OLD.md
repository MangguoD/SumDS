# 简单的使用教程
如果使用GLM，请更换GLM的虚拟环境
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
lv1输出可用度在86.8%（不准确），速度为499条/3.39h
输出图片见DS_lv1_output_500

2025/3/22
lv2:1. 增加了约束力(也许是副作用？）
    2. 增加了输入提示模版，增强输出结构性，并且能够优化速度（不做retry的话平均10s/条）
    3. 删除了眼底以及并发症的总结（因为大部分原数据都没有）
    4. 增加了输出后对规范性检测的内容，设置了宽容度，并且多次检测，提升输出质量和可用性（导致处理速度变长的主要原因）
        宽容度输出效果如下：
            [INVALID OUTPUT] 第 1 次尝试：结构不足 4 段
            第 2 次尝试成功：结构至少有 3 段
        也就是说，随着失败次数增加，对结构段落要求会逐渐降低
    3/23日志：降低了要求，即使最后结果错误也会输出保存
    5. 能够保存错误输出记录，就算没有正确输出四条也会保存
    6. 增加了flash-attention支持的判断，自动选择
    7. 增加了批处理功能，在显存允许的情况下大幅增加处理效率
lv2_v1输出可用度在80%，速度在1168条/4.68h
输出图片见：DS_lv2_output_v1
lv2的不可用输出无论是在数量上还是位置上都和GLM的loss曲线相似，推测是这些位置有着过长的信息，超过模型的token或者思维能力的上限

2025/3/24
lv2_text_cut:1.增加了文本截断，DS-7b的最大能力是约16000，设置截断为8000，new_token为1200，调整出了冗余空间
             2.调整了显存释放逻辑，现在释放时间更久，长时间运行更稳定
             3.现在全局默认左padding
             4.再次强化了提示词逻辑，保证模型规范输出的情况下不会丢失输入信息（自动截断）
             5.迭代了后处理结构，现在能够删除
                 - think部分
                 - 【结构模版】（用于提示模型输出）
               并且针对一些常见错误输出，会直接返厂维修，也会进行清洗
lv2_text_cut的准确率在92.9%并且还有进一步提升空间
输出图片见：DS_lv2_cut_output_V1