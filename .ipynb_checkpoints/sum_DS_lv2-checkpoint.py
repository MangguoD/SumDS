import time
import torch
import pandas as pd
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

print("开始运行程序...")

# 模型路径和设备
model_name = "../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"  # V100 不支持 flash-attention
).eval()

# 模板提示生成函数
def query_llm(prompt):
    """调用 LLM 并返回处理后的响应文本。"""
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

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0
        )

    output_ids = generated[0][inputs.input_ids.shape[1]:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    del inputs, generated, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    return process_response(output)

# 删除模型输出中多余部分
def process_response(text):
    marker = "</think>"
    pos = text.find(marker)
    if pos == -1:
        return text
    end_index = pos + len(marker)
    while end_index < len(text) and text[end_index] == "\n":
        end_index += 1
    return text[end_index:]

# 检查输出结构是否完整
def is_valid_output(output):
    required_sections = [
        "1. 血糖控制",
        "2. 血压管理",
        "3. 眼底以及并发症",
        "4. 依从性与监测问题",
        "5. 生活方式"
    ]
    return all(section in output for section in required_sections)

# 读取数据
df = pd.read_excel("./input/joined_condition_500.xlsx")
if 'patient_condition' not in df.columns:
    raise ValueError("Excel 文件中未找到 'patient_condition' 列，请检查文件格式。")

# 加载已处理数据（若存在）
try:
    df_out = pd.read_excel("./output/joined_condition_summarized_500_DS.xlsx")
    processed_indices = set(df_out.index[df_out['response'].notna()])
    print(f"检测到已有 {len(processed_indices)} 条已处理记录，将跳过这些记录。")
except FileNotFoundError:
    df_out = df.copy()
    df_out['response'] = ""
    processed_indices = set()

error_records = []
total = len(df)
start_time = time.time()

# 主循环处理数据
for i, row in df.iterrows():
    if i in processed_indices:
        continue

    prompt = str(row['patient_condition'])

    try:
        response = query_llm(prompt)
        if is_valid_output(response):
            df_out.at[i, 'response'] = response
        else:
            error_message = "[INVALID OUTPUT] 缺失结构段落"
            df_out.at[i, 'response'] = error_message
            error_records.append((i, prompt, error_message))

    except Exception as e:
        error_message = f"[ERROR] {e}"
        df_out.at[i, 'response'] = error_message
        error_records.append((i, prompt, error_message))

    # 显示进度
    completed = i + 1
    elapsed = time.time() - start_time
    print(f"DeepSeek_lv2已完成 {completed}/{total} 条，耗时 {elapsed:.2f} 秒")

    # 每 5 条保存一次中间文件
    if completed % 5 == 0:
        df_out.to_excel("./output/joined_condition_summarized_500_DS.xlsx", index=False)

# 最终保存输出
df_out.to_excel("./output/joined_condition_summarized_500_DS.xlsx", index=False)

# 保存错误记录
if error_records:
    df_error = pd.DataFrame(error_records, columns=["index", "prompt", "error"])
    df_error.to_excel("./output/deepseek_errors.xlsx", index=False)
    print(f"有 {len(error_records)} 条数据处理失败或结构不完整，详见 deepseek_errors.xlsx")

# 输出确定
print("####所有任务处理完成####")