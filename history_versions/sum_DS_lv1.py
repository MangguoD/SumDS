# Author: MangguoD

import time
import torch
import pandas as pd
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

print("开始运行程序...")

# 模型加载
model_name = "../../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2" # 效率，需要先编译flash-attention的环境（py3.10) V100不支持
#    attn_implementation="eager"  # 使用默认注意力机制
 ).eval()

# LLM 询问函数 "填空式模板" "绝对语气" “标签名”
def query_llm(prompt):
    """发送 prompt 给模型，并返回结构化整理结果。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名经验丰富的医疗文档整理专家。"
                "禁止输出任何解释、分析、推理、注释或中间过程，"
                "你的任务是严格按照用户提供的结构模板输出整理结果。"
            )
        },
        {
            "role": "user",
            "content": (
                f"请根据以下病情描述内容整理出结构化结果：\n\n"
                f"{prompt}\n\n"
                "按照如下结构输出（请不要更改格式/标题，不要添加前言结尾）：\n"
                "1. 血糖控制\n"
                "   ○ 空腹血糖波动情况：\n"
                "   ○ 餐后血糖波动情况：\n"
                "   ○ 糖化血红蛋白（HbA1c）：\n"
                "   ○ 症状：\n"
                "2. 血压管理\n"
                "3. 眼底以及并发症\n"
                "4. 依从性与监测问题\n"
                "5. 生活方式\n"
                "   ○ 饮食：\n"
                "   ○ 运动：\n"
                "   ○ 体重变化："
            )
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,  # 控制输出长度
            do_sample=False       # 按模板输出
        )

    output_ids = generated[0][inputs.input_ids.shape[1]:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    # 释放显存
    del inputs, generated, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    return process_response(output)

# 处理生成文本 删除think部分
def process_response(text):
    marker = "</think>"
    pos = text.find(marker)
    if pos == -1:
        return text
    end_index = pos + len(marker)
    while end_index < len(text) and text[end_index] == "\n":
        end_index += 1
    return text[end_index:]

# 读取并处理数据
df = pd.read_excel("../input/joined_condition_500.xlsx")
if 'patient_condition' not in df.columns:
    raise ValueError("Excel 文件中未找到 'patient_condition' 列，请检查文件格式。")

# 若存在中途保存的结果，加载并跳过已完成部分
try:
    df_out = pd.read_excel("../output/joined_condition_summarized_500_DS.xlsx")
    processed_indices = set(df_out.index[df_out['response'].notna()])
    print(f"检测到已有 {len(processed_indices)} 条已处理记录，将跳过这些记录。")
except FileNotFoundError:
    df_out = df.copy()
    df_out['response'] = ""
    processed_indices = set()

total = len(df)
start_time = time.time()

# 主循环
for i, row in df.iterrows():
    if i in processed_indices:
        continue

    prompt = str(row['patient_condition'])

    try:
        response = query_llm(prompt)
        processed = process_response(response)
        df_out.at[i, 'response'] = processed
    except Exception as e:
        print(f"第 {i} 条处理出错：{e}")
        df_out.at[i, 'response'] = f"[ERROR] {e}"

    # 显示进度
    completed = i + 1
    elapsed = time.time() - start_time
    print(f"DeepSeek_lv1已完成 {completed}/{total} 条，耗时 {elapsed:.2f} 秒")

    # 每 5 条保存一次
    if completed % 5 == 0:
        df_out.to_excel("../output/joined_condition_summarized_500_DS.xlsx", index=False)

# 最终保存
df_out.to_excel("../output/joined_condition_summarized_500_DS.xlsx", index=False)
print("#####FINISHIED!#####")