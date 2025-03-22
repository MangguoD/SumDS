import time
import torch
import pandas as pd
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

print("开始运行程序...")

# 模型路径和设备
model_name = "../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attn_implementation():
    capability = torch.cuda.get_device_capability()
    return "flash_attention_2" if capability[0] >= 8 else "eager"

attn_impl = get_attn_implementation()

# 模型加载
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto" if device.type == "cuda" else None,
    attn_implementation=attn_impl if device.type == "cuda" else None
).to(device).eval()

# 模板提示生成函数（支持动态 max_new_tokens）
def query_llm(prompt, max_new_tokens=1024):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一名经验丰富的医疗文档结构化处理专家。请你参考以下“示例输入+输出”，"
                "并严格按照“结构模板”对【实际输入】进行结构化整理，缺失项请写 none，不得更改格式。"
            )
        },
        {
            "role": "user",
            "content": (
                "【示例输入1】\n"
                "患者男性，62岁，BMI 30.12，身高168cm，体重85kg。血压为130/85 mmHg，"
                "空腹血糖多次在7.0~7.7 mmol/L之间，餐后血糖为10.7 mmol/L。"
                "无糖尿病症状，无低血糖反应。饮食控制良好，近期有换用二甲双胍的打算。"
                "体重无明显变化，无具体运动记录，未见眼底并发症描述。\n\n"
                "【示例输出1】\n"
                "1. 血糖控制\n"
                "   ○ 空腹血糖波动情况：空腹血糖稳定在7左右，波动较小。\n"
                "   ○ 餐后血糖波动情况：餐后血糖稳定在10.7mmol/L左右，波动不大。\n"
                "   ○ 糖化血红蛋白（HbA1c）：未提及具体数值。\n"
                "   ○ 症状：无糖尿病症状。\n"
                "2. 血压管理\n"
                "   ○ 收缩压：130mmHg，正常。\n"
                "   ○ 舒张压：85mmHg，正常。\n"
                "3. 依从性与监测问题\n"
                "   ○ 用药情况：考虑换用二甲双胍。\n"
                "   ○ 饮食控制：良好。\n"
                "   ○ 体重变化：体重稳定，无明显变化。\n"
                "4. 生活方式\n"
                "   ○ 饮食：良好。\n"
                "   ○ 运动：无记录。\n"
                "   ○ 体重变化：无明显变化。\n\n"

                "【示例输入2】\n"
                "患者女性，74岁，BMI 24.22，身高160cm，体重62kg。血压128/77mmHg，"
                "空腹血糖波动在6.5~8.5 mmol/L，餐后血糖波动在8.3~10.3 mmol/L。"
                "多次记录显示依从性差、未监测餐后血糖，无糖尿病症状和低血糖反应，"
                "饮食、运动、体重记录缺失，未见眼底并发症描述。\n\n"
                "【示例输出2】\n"
                "1. 血糖控制\n"
                "   ○ 空腹血糖波动情况：波动在6.5~8.5mmol/L，较大。\n"
                "   ○ 餐后血糖波动情况：波动在8.3~10.3mmol/L，较大。\n"
                "   ○ 糖化血红蛋白（HbA1c）：none\n"
                "   ○ 症状：无糖尿病症状。\n"
                "2. 血压管理\n"
                "   ○ 收缩压：128mmHg\n"
                "   ○ 舒张压：77mmHg\n"
                "3. 依从性与监测问题\n"
                "   ○ 依从性：多次记录为依从性差，血糖不稳定。\n"
                "4. 生活方式\n"
                "   ○ 饮食：none\n"
                "   ○ 运动：none\n"
                "   ○ 体重变化：none\n\n"

                "【结构模板】\n"
                "请按照以下结构输出，缺失项写 none，不得更改标题：\n"
                "1. 血糖控制\n"
                "   ○ 空腹血糖波动情况：\n"
                "   ○ 餐后血糖波动情况：\n"
                "   ○ 糖化血红蛋白（HbA1c）：\n"
                "   ○ 症状：\n"
                "2. 血压管理\n"
                "3. 依从性与监测问题\n"
                "4. 生活方式\n"
                "   ○ 饮食：\n"
                "   ○ 运动：\n"
                "   ○ 体重变化：\n\n"
                "【实际输入】\n"
                f"{prompt}"
            )
        }
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated[0][inputs.input_ids.shape[1]:]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    del inputs, generated, output_ids
    torch.cuda.empty_cache()
    gc.collect()

    return process_response(output) # 输出部分（移除think模块）

def is_valid_output(output, min_sections=4):
    required_sections = [
        "1. 血糖控制",
        "2. 血压管理",
        "3. 依从性与监测问题",
        "4. 生活方式"
    ]
    matched = [s for s in required_sections if s in output]
    return len(matched) >= min_sections

# 处理think部分
def process_response(text):
    marker = "</think>"
    pos = text.find(marker)
    if pos == -1:
        return text
    end_index = pos + len(marker)
    while end_index < len(text) and text[end_index] == "\n":
        end_index += 1
    return text[end_index:]

# 加载数据
df = pd.read_excel("./input/joined_condition_500.xlsx")
if 'patient_condition' not in df.columns:
    raise ValueError("Excel 文件中未找到 'patient_condition' 列，请检查文件格式。")

try:
    df_out = pd.read_excel("./output/joined_condition_summarized_500_DSlv2.xlsx")
    processed_indices = set(df_out.index[df_out['response'].notna()])
    print(f"检测到已有 {len(processed_indices)} 条已处理记录，将跳过这些记录。")
except FileNotFoundError:
    df_out = df.copy()
    df_out['response'] = ""
    processed_indices = set()

error_records = []
total = len(df)
start_time = time.time()
max_retries = 3
token_steps = [1024, 1536, 2048]
section_steps = [4, 3, 2]

for i, row in df.iterrows():
    if i in processed_indices:
        continue

    prompt = str(row['patient_condition'])

    for attempt in range(max_retries):
        try:
            max_tokens = token_steps[attempt]
            min_sections = section_steps[attempt]

            response = query_llm(prompt, max_new_tokens=max_tokens)
            if is_valid_output(response, min_sections):
                print(f"第 {attempt+1} 次尝试成功：结构至少有 {min_sections} 段")
                df_out.at[i, 'response'] = response
                break
            else:
                print(f"[INVALID OUTPUT] 第 {attempt+1} 次尝试：结构不足 {min_sections} 段")
                if attempt == max_retries - 1:
                    err = f"[INVALID OUTPUT][RETRY_FAILED] 共尝试 {max_retries} 次失败"
                    df_out.at[i, 'response'] = err
                    error_records.append((i, prompt, err))

        except Exception as e:
            print(f"[ERROR] 第 {attempt+1} 次尝试失败：{e}")
            if attempt == max_retries - 1:
                err = f"[ERROR][RETRY_FAILED] {e}"
                df_out.at[i, 'response'] = err
                error_records.append((i, prompt, err))

        torch.cuda.empty_cache()
        gc.collect()

    completed = i + 1
    elapsed = time.time() - start_time
    print(f"DeepSeek_lv2已完成 {completed}/{total} 条，耗时 {elapsed:.2f} 秒")

    if completed % 5 == 0:
        df_out.to_excel("./output/joined_condition_summarized_500_DSlv2.xlsx", index=False)

df_out.to_excel("./output/joined_condition_summarized_500_DSlv2.xlsx", index=False)

if error_records:
    df_error = pd.DataFrame(error_records, columns=["index", "prompt", "error"])
    df_error.to_excel("./output/deepseek_errors.xlsx", index=False)
    print(f"有 {len(error_records)} 条数据处理失败或结构不完整，详见 deepseek_errors.xlsx")

print("#### 所有任务处理完成 ####")
