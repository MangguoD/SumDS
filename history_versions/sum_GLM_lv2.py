# Author: MangguoD

import time
import torch
import pandas as pd
import gc
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore") # 忽略下面的警告

print("开始运行程序...")

# 模型路径和设备
model_name = "../../autodl-tmp/glm-4-9b-chat" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

# 模板提示生成函数
def build_prompt(prompt, mode="strict"):
    if mode == "strict":
        return [
            {
                "role": "system",
                "content": (
                    "你是一名经验丰富的医疗文档结构化处理专家，请你严格参照【结构模板】，"
                    "参考【实例输入】对【实际输入】进行结构化整理，缺失项请写 none，不得更改格式，如果有回访，将波动情况改为范围值"
                    "必须完整输出所有 4 个结构段，缺失任何一段即为不合格。不得省略或压缩任何结构段。"
                )
            },
            {
                "role": "user",
                "content": (
                    "【示例输入】\n"
                    "患者男性，62岁，BMI 30.12，身高168cm，体重85kg。血压为130/85 mmHg，"
                    "空腹血糖多次在7.0~7.7 mmol/L之间，餐后血糖为10.7 mmol/L。"
                    "无糖尿病症状，无低血糖反应。饮食控制良好，近期有换用二甲双胍的打算。"
                    "体重无明显变化，无具体运动记录\n\n"
                    
                    "【示例输出】\n"
                    "### 1. 血糖控制\n"
                    "- 空腹血糖波动情况：空腹血糖稳定在7左右，波动较小。\n"
                    "- 餐后血糖波动情况：餐后血糖稳定在10.7mmol/L左右，波动不大。\n"
                    "- 症状：无糖尿病症状。\n"
                    "### 2. 血压管理\n"
                    "- 收缩压：130mmHg，正常。\n"
                    "- 舒张压：85mmHg，正常。\n"
                    "### 3. 依从性与监测问题\n"
                    "- 依从性：none\n"
                    "- 用药情况：考虑换用二甲双胍。\n"
                    "### 4. 生活方式\n"
                    "- 饮食：良好。\n"
                    "- 运动：无记录。\n"
                    "- 体重变化：体重变化：无明显变化。\n\n"
                    
                    "【结构模板】\n"
                    "请严格按照以下结构输出，缺失项写 none，不得更改标题：\n"
                    "### 1. 血糖控制\n"
                    "- 空腹血糖波动情况：\n"
                    "- 餐后血糖波动情况：\n"
                    "- 症状：\n"
                    "### 2. 血压管理\n"
                    "### 3. 依从性与监测问题\n"
                    "- 依从性：\n"
                    "- 用药情况：\n"
                    "### 4. 生活方式\n"
                    "- 饮食：\n"
                    "- 运动：\n"
                    "- 体重变化：\n\n"
                    
                    "【实际输入】\n"
                    f"{prompt}"
                )
            }
        ]
    # 宽容模式
    else:
        return [
            {
                "role": "user",
                "content": (
                    "请根据以下文本内容，尽可能完整地提取结构化信息，"
                    "输出格式如下：\n"
                    "### 1. 血糖控制\n"
                    "- 空腹血糖波动情况：\n"
                    "- 餐后血糖波动情况：\n"
                    "- 症状：\n"
                    "### 2. 血压管理\n"
                    "### 3. 依从性与监测问题\n"
                    "### 4. 生活方式\n"
                    "- 饮食：\n"
                    "- 运动：\n"
                    "- 体重变化：\n\n"
                
                    f"{prompt}"
                )
            }
        ]

# 处理响应中 think 模块（如果有）
def process_response(text):
    marker = "</think>"
    pos = text.find(marker)
    if pos == -1:
        return text
    end_index = pos + len(marker)
    while end_index < len(text) and text[end_index] == "\n":
        end_index += 1
    return text[end_index:]

# 判断结构完整性
def is_valid_output(output, min_sections=4):
    required_sections = [
        "1. 血糖控制",
        "2. 血压管理",
        "3. 依从性与监测问题",
        "4. 生活方式"
    ]
    matched = [s for s in required_sections if s in output]
    matched_count = len(matched)
    return matched_count, matched_count >= min_sections

# GLM 推理调用
def query_llm_batch(prompts, max_new_tokens=512, fallback=False):
    mode = "fallback" if fallback else "strict"
    messages_list = [build_prompt(p, mode) for p in prompts]

    full_prompts = []
    for messages in messages_list:
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) # promot拼接
        full_prompts.append(full_prompt)

    inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,  # 禁止连续3词重复（可调）
        )

    results = []
    for i in range(len(full_prompts)):
        output_ids = outputs[i][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        results.append(process_response(text))

    return results

# 加载数据
df = pd.read_excel("../input/joined_condition_500.xlsx")
if 'patient_condition' not in df.columns:
    raise ValueError("Excel 文件中未找到 'patient_condition' 列，请检查文件格式。")

try:
    df_out = pd.read_excel("../output/joined_condition_summarized_500_GLM3.xlsx")
    processed_indices = set(df_out.index[df_out['response'].notna()])
    print(f"检测到已有 {len(processed_indices)} 条已处理记录，将跳过这些记录。")
except FileNotFoundError:
    df_out = df.copy()
    df_out['response'] = ""
    processed_indices = set()

# 处理流程控制
error_records = []
total = len(df)
start_time = time.time()
max_retries = 2
token_steps = [1024, 2048]
section_steps = [3, 2]

batch_size = 10  # 批处理流程，控制大小小心爆显存
batched_indices = [df.index[i:i+batch_size] for i in range(0, len(df), batch_size)]

for batch_ids in batched_indices:
    unprocessed_ids = [i for i in batch_ids if i not in processed_indices]
    if not unprocessed_ids:
        continue

    prompts = [str(df.loc[i, 'patient_condition']) for i in unprocessed_ids]

    # 容忍处理流程
    try:
        responses = query_llm_batch(prompts, max_new_tokens=1024)
        for j, i in enumerate(unprocessed_ids):
            resp = responses[j]
            matched_count, is_valid = is_valid_output(resp, min_sections=4)
            df_out.at[i, 'response'] = resp
            if is_valid:
                df_out.at[i, 'status'] = "FULL"
            elif matched_count >= 2:
                df_out.at[i, 'status'] = f"PARTIAL_{matched_count}"
            else:
                df_out.at[i, 'status'] = "FAIL"
                error_records.append((i, prompts[j], resp))
    except Exception as e:
        print(f"[BATCH ERROR] 批次处理失败：{e}")
        for j, i in enumerate(unprocessed_ids):
            df_out.at[i, 'response'] = f"[BATCH ERROR] {e}"
            df_out.at[i, 'status'] = "ERROR"
            error_records.append((i, prompts[j], str(e)))

    completed = max(batch_ids) + 1
    elapsed = time.time() - start_time
    print(f"ChatGLM 批处理已完成 {completed}/{len(df)} 条，耗时 {elapsed:.2f} 秒")

    if completed % 20 == 0:
        df_out.to_excel("../output/joined_condition_summarized_batch.xlsx", index=False)
        torch.cuda.empty_cache()
        gc.collect()
    # 显存释放，不然一定会爆的
    torch.cuda.empty_cache()
    gc.collect()

# 最终保存
df_out.to_excel("../output/joined_condition_summarized_batch.xlsx", index=False)

if error_records:
    df_error = pd.DataFrame(error_records, columns=["index", "prompt", "error"])
    df_error.to_excel("./output/chatglm_batch_errors.xlsx", index=False)
    print(f"有 {len(error_records)} 条数据处理失败，详见 chatglm_batch_errors.xlsx")

print("#### 批处理结构化任务完成 ####")