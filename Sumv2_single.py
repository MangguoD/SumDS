# 这个是DS模拟用户输入进行处理的单条处理代码。
import time
import torch
import pandas as pd
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from torch.nn.functional import pad  # 左padding库

warnings.filterwarnings("ignore") # 忽略警告,无视风险,继续运行

print("开始运行程序...")

# 模型路径和设备
model_name = "../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attn_implementation():
    capability = torch.cuda.get_device_capability()
    return "flash_attention_2" if capability[0] >= 8 else "eager"

attn_impl = get_attn_implementation()

# 模型加载 左截断
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left"  # 必须在这里设置
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto" if device.type == "cuda" else None,
    attn_implementation=attn_impl if device.type == "cuda" else None
).to(device).eval()

# 构造提示词
def build_prompt_tokens(raw_text, tokenizer, max_input_tokens):
    """
    构建完整 prompt 并自动按 token 截断 raw_text，以确保总长不超过 max_input_tokens
    返回: input_ids（List[int]）
    """

    # System prompt
    system_content = (
        "你是医疗文档结构抽取工具。\n"
        "你的任务是严格按照用户提供的结构模板输出整理结果。"
        "对频繁出现的数值，仅提取：\n"
        "- 常见值 / 平均值\n"
        "- 波动范围（最低 ~ 最高）\n"
        "- 波动幅度（最高 - 最低）\n"
        "缺失信息填“未提及”。\n"
        "禁止输出：多余说明、分析、表格、注释、格式提示、模板重复。\n"
    )
    system_ids = tokenizer(system_content, return_tensors=None, add_special_tokens=False)["input_ids"]

    # 构造结构化模板前缀（包含 raw_text 的前缀）
    prefix = "请根据以下病情描述内容整理出结构化结果：\n\n"
    prefix_ids = tokenizer(prefix, return_tensors=None, add_special_tokens=False)["input_ids"]

    # 构造结构模板后缀
    structure_suffix = (
        "\n请将提取结果仅填入以下【结构模板】中。\n"

        "【结构模板】：\n\n"
        "1. **血糖控制**\n"
        "- 空腹血糖波动情况：\n"
        " - 平均值：\n"
        " - 波动范围：\n"
        " - 波动幅度：\n"
        "- 餐后血糖波动情况：\n"
        " - 平均值：\n"
        " - 波动范围：\n"
        " - 波动幅度：\n"
        "- 症状：\n"
        "2. **血压管理**\n"
        "- 收缩压：\n"
        "- 舒张压：\n"
        "- 空腹血压：\n"
        "- 餐后血压：\n"
        "3. **依从性与监测问题**\n"
        "- 依从性：\n"
        "- 用药情况：\n"
        "4. **生活方式**\n"
        "- 饮食：\n"
        "- 运动：\n"
        "- 体重变化：\n\n"

        "\n【仅填写以上模板字段，输出到此为止。】"
    )
    suffix_ids = tokenizer(structure_suffix, return_tensors=None, add_special_tokens=False)["input_ids"]

    # 给 raw_text 剩下的 token budget
    reserved = len(system_ids) + len(prefix_ids) + len(suffix_ids)
    available_budget = max_input_tokens - reserved
    if available_budget <= 0:
        raise ValueError(f"[错误] max_input_tokens={max_input_tokens} 太小，不足以容纳模板固定部分")

    # 截断 raw_text
    raw_text_ids = tokenizer(raw_text, return_tensors=None, add_special_tokens=False)["input_ids"]
    if len(raw_text_ids) > available_budget:
        raw_text_ids = raw_text_ids[:available_budget]

    # 拼接成完整 input_ids
    full_ids = system_ids + prefix_ids + raw_text_ids + suffix_ids
    return full_ids


# 检查输出结构完整性
def is_valid_output(output, min_sections=4):
    required_sections = [
        "**血糖控制**",
        "**血压管理**",
        "**依从性与监测问题**",
        "**生活方式**"
    ]
    matched = [s for s in required_sections if s in output]
    return len(matched), len(matched) >= min_sections


# 后处理部分
def process_response(text):
    # 移除 <think> 部分
    marker = "</think>"
    pos = text.find(marker)
    if pos != -1:
        end_index = pos + len(marker)
        while end_index < len(text) and text[end_index] == "\n":
            end_index += 1
        text = text[end_index:]

    # 删除所有出现的“【结构模版】”或“【结构模板】”
    text = text.replace("【结构模版】", "").replace("【结构模板】", "")

    # 去除模型尾部意外重复模板
    template_signals = [
        "请根据提供的病情描述内容整理出结构化结果",
        "结构模版：",
        "请将提取结果"
    ]
    for key in template_signals:
        if key in text:
            text = text.split(key)[0].strip()

    # 清洗重复 checklist 段
    lines = text.splitlines()
    deduped = []
    seen = set()
    for line in lines:
        if line.strip() not in seen:
            deduped.append(line)
            seen.add(line.strip())

    return "\n".join(deduped)

    
# 单条处理部分
def query_llm_single(raw_text, max_new_tokens=2200):
    MAX_INPUT_TOKENS = 8000
    MODEL_MAX_LENGTH = 16384

    full_ids = build_prompt_tokens(raw_text, tokenizer, MAX_INPUT_TOKENS)
    total_len = len(full_ids)

    if total_len + max_new_tokens > MODEL_MAX_LENGTH:
        allowed = MODEL_MAX_LENGTH - max_new_tokens
        print(f"[硬性截断] prompt 总 token 数为 {total_len}，超过最大值 {MODEL_MAX_LENGTH - max_new_tokens}，截为前 {allowed} 个 token")
        full_ids = full_ids[:allowed]

    input_ids = torch.tensor(full_ids)
    attention_mask = torch.ones_like(input_ids)

    # 左 padding（这里只处理单条，因此不用 stack）
    max_len = input_ids.size(0)
    input_ids_padded = pad(input_ids, (0, 0), value=tokenizer.pad_token_id).unsqueeze(0)
    attention_mask_padded = pad(attention_mask, (0, 0), value=0).unsqueeze(0)

    encodings = {
        "input_ids": input_ids_padded.to(device),
        "attention_mask": attention_mask_padded.to(device)
    }

    with torch.no_grad():
        outputs = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = (encodings["attention_mask"] == 1).sum(dim=1)[0]
    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return process_response(text)


# 输入部分
print("\n请输入患者病情描述（输入 q 回车退出）：")
while True:
    raw = input("\n📝 患者情况：\n")
    if raw.lower() in ['q', 'quit', 'exit']:
        break
    try:
        result = query_llm_single(raw)
        matched_count, is_valid = is_valid_output(result)
        print("\n🧾 模型输出：\n")
        print(result)
        print(f"\n✅ 匹配字段数：{matched_count}，状态：{'FULL' if is_valid else 'PARTIAL' if matched_count >= 2 else 'FAIL'}")
    except Exception as e:
        print(f"[❌ 错误] 处理失败：{e}")

    # 显存清理
    torch.cuda.empty_cache()
    gc.collect()


# 加载数据
df = pd.read_excel("./input/joined_condition.xlsx")
try:
    df_out = pd.read_excel("./output/DS_lv2_cut/joined_DSlv2_text_cut.xlsx")
    processed_indices = set(df_out.index[df_out['response'].notna()])
    print(f"检测到已有 {len(processed_indices)} 条已处理记录，将跳过这些记录。")
except FileNotFoundError:
    df_out = df.copy()
    df_out['response'] = ""
    df_out['status'] = ""
    processed_indices = set()

batch_size = 15 #控制批次大小
error_records = []
total = len(df)
start_time = time.time()

# 标记输出情况，保留错误或截断输出，方便统计
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
    # 释放显存（这段真的很重要）
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1) #确保显存释放干净（秒），如果还有爆显存的情况就加到1.25
    print(f"DeepSeek_lv2已完成 {completed}/{total} 条，耗时 {elapsed:.2f} 秒")

    # 在显存大于 40GB 时释放显存
#    if torch.cuda.is_available():
#        current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
#        if current_mem > 40:
#            print(f"[显存释放] 当前占用约 {current_mem:.2f} GB，执行释放...")
#            torch.cuda.empty_cache()
#            gc.collect()

    if completed % 15 == 0:
        df_out.to_excel("./output/DS_lv2_cut/joined_DSlv2_text_cut.xlsx", index=False)


df_out.to_excel("./output/DS_lv2_cut/joined_DSlv2_text_cut.xlsx", index=False)

if error_records:
    df_error = pd.DataFrame(error_records, columns=["index", "prompt", "error"])
    df_error.to_excel("./output/DS_lv2_cut/deepseek_errors_cut.xlsx", index=False)
    print(f"有 {len(error_records)} 条数据处理失败，详见 deepseek_errors_batch8.xlsx")

print("#### 所有任务处理完成 ####")