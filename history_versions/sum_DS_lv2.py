# Author: MangguoD

import time
import torch
import pandas as pd
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

print("开始运行程序...")

# 模型路径和设备
model_name = "../../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"
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

# 构造提示词
def build_prompt(prompt, mode="strict"):
    if mode == "strict":
        return [
        {
            "role": "system",
            "content": (
                "你是一位医疗文档结构整理机器人，只能根据原文内容提取信息填入固定结构模板，"
                "禁止进行归纳、总结、整合、提炼、推理、发挥。不得合并多条内容，也不得将数值列表化。"
                "你的任务是严格按照用户提供的结构模板输出整理结果。"
            )
        },
        {
            "role": "user",
            "content": (
                f"请根据以下病情描述内容整理出结构化结果：\n\n"
                f"{prompt}\n\n"
                "按照如下结构输出（请不要更改格式/标题，不要添加前言结尾）：\n"
                "1. **血糖控制**\n"
                "- 空腹血糖波动情况：\n"
                "- 餐后血糖波动情况：\n"
                "- 症状：\n"
                "2. **血压管理**\n"
                "3. **依从性与监测问题**\n"
                "- 依从性：\n"
                "- 用药情况：\n"
                "4. **生活方式**\n"
                "- 饮食：\n"
                "- 运动：\n"
                "- 体重变化：\n\n"
            )
        }
        ]
    else:
        return [
            {
                "role": "user",
                "content": (
                    "请根据以下文本内容，尽可能完整地提取结构化信息，"
                    "输出格式如下：\n"
                    "1. **血糖控制**\n"
                    "- 空腹血糖波动情况：\n"
                    "- 餐后血糖波动情况：\n"
                    "- 症状：\n"
                    "2. **血压管理**\n"
                    "3. **依从性与监测问题**\n"
                    "4. **生活方式**\n"
                    "- 饮食：\n"
                    "- 运动：\n"
                    "- 体重变化：\n\n"
                    f"{prompt}"
                )
            }
        ]

# 检查输出结构完整性
def is_valid_output(output, min_sections=4):
    required_sections = [
        "1. **血糖控制**",
        "2. **血压管理**",
        "3. **依从性与监测问题**",
        "4. **生活方式**"
    ]
    matched = [s for s in required_sections if s in output]
    return len(matched), len(matched) >= min_sections

# 移除 <think> 部分
def process_response(text):
    marker = "</think>"
    pos = text.find(marker)
    if pos == -1:
        return text
    end_index = pos + len(marker)
    while end_index < len(text) and text[end_index] == "\n":
        end_index += 1
    return text[end_index:]

# 批处理主函数
def query_llm_batch(prompts, max_new_tokens=1200):
    MAX_INPUT_TOKENS = 8000  # 设置最大输入Token长度

    prompt_list = [tokenizer.apply_chat_template(build_prompt(p), tokenize=False, add_generation_prompt=True) for p in prompts]

    input_ids_list = []
    attention_mask_list = []

    for idx, prompt in enumerate(prompt_list):
        tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"][0]

        if input_ids.shape[0] > MAX_INPUT_TOKENS:
            print(f"[截断警告] 第 {idx+1} 条 prompt 超过 {MAX_INPUT_TOKENS} 个 tokens，已截断为前 {MAX_INPUT_TOKENS} tokens")
            input_ids = input_ids[:MAX_INPUT_TOKENS]

        attention_mask = torch.ones_like(input_ids)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    # Padding 成 batch
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    encodings = {
        "input_ids": input_ids_padded.to(device),
        "attention_mask": attention_mask_padded.to(device)
    }

    with torch.no_grad():
        outputs = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
#            top_p=0.9,
#            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
#            repetition_penalty=1.15,
#            no_repeat_ngram_size=3,
        )

    results = []
    input_lengths = (encodings["attention_mask"] == 1).sum(dim=1)
    for i, input_len in enumerate(input_lengths):
        gen_ids = outputs[i][input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(process_response(text))
    return results

# 加载数据
df = pd.read_excel("../input/joined_condition.xlsx")
try:
    df_out = pd.read_excel("../output/joined_condition_DSlv2_batch8.xlsx")
    processed_indices = set(df_out.index[df_out['response'].notna()])
    print(f"检测到已有 {len(processed_indices)} 条已处理记录，将跳过这些记录。")
except FileNotFoundError:
    df_out = df.copy()
    df_out['response'] = ""
    df_out['status'] = ""
    processed_indices = set()

batch_size = 8
error_records = []
total = len(df)
start_time = time.time()

for start_idx in range(0, total, batch_size):
    end_idx = min(start_idx + batch_size, total)
    batch_indices = [i for i in range(start_idx, end_idx) if i not in processed_indices]
    if not batch_indices:
        continue

    batch_prompts = [str(df.loc[i, 'patient_condition']) for i in batch_indices]

    try:
        batch_outputs = query_llm_batch(batch_prompts)
        for i, output in zip(batch_indices, batch_outputs):
            matched_count, is_valid = is_valid_output(output)
            df_out.at[i, 'response'] = output
            if is_valid:
                df_out.at[i, 'status'] = "FULL"
            elif matched_count >= 2:
                df_out.at[i, 'status'] = f"PARTIAL_{matched_count}"
            else:
                df_out.at[i, 'status'] = "FAIL"
                error_records.append((i, batch_prompts[i - start_idx], output))
    except Exception as e:
        for i in batch_indices:
            df_out.at[i, 'response'] = f"[ERROR] {e}"
            df_out.at[i, 'status'] = "ERROR"
            error_records.append((i, df.loc[i, 'patient_condition'], str(e)))

    completed = end_idx
    elapsed = time.time() - start_time
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.5) #确保显存释放干净
    print(f"DeepSeek_lv2已完成 {completed}/{total} 条，耗时 {elapsed:.2f} 秒")

    # 在显存大于 40GB 时释放显存
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        if current_mem > 40:
            print(f"[显存释放] 当前占用约 {current_mem:.2f} GB，执行释放...")
            torch.cuda.empty_cache()
            gc.collect()

    if completed % 8 == 0:
        df_out.to_excel("../output/joined_condition_DSlv2_batch8.xlsx", index=False)


df_out.to_excel("../output/joined_condition_DSlv2_batch8.xlsx", index=False)

if error_records:
    df_error = pd.DataFrame(error_records, columns=["index", "prompt", "error"])
    df_error.to_excel("../output/deepseek_errors_batch10.xlsx", index=False)
    print(f"有 {len(error_records)} 条数据处理失败，详见 deepseek_errors_batch10.xlsx")

print("#### 所有任务处理完成 ####")