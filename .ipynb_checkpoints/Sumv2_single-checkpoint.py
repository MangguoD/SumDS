# Author: MangguoD
# Sumv2_single.py for workflow
import os
import time
import torch
import gc
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from torch.nn.functional import pad  # 左padding库

warnings.filterwarnings("ignore")  # 忽略警告,无视风险,继续运行

# 配置日志，将每次调用的输入及响应写入 workflow_input.log 文件
logging.basicConfig(
    filename="workflow_input.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 模型设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attn_implementation():
    capability = torch.cuda.get_device_capability()
    return "flash_attention_2" if capability[0] >= 8 else "eager"

attn_impl = get_attn_implementation()

# 全局变量：模型和 tokenizer（在加载后赋值）
model = None
tokenizer = None

# 封装函数：加载 DeepSeek-7B 模型和 tokenizer（仅从本地路径）
def load_model_and_tokenizer_7b(model_path: str = None):
    """
    加载 DeepSeek-7B 模型及对应的 tokenizer。
    参数:
      model_path: 模型本地路径目录。如果为 None，则使用默认路径。
    返回:
      (model, tokenizer) 元组。
    """
    global model, tokenizer
    # 默认模型路径
    if model_path is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"[错误] 模型路径不存在：{model_path}")
    # 从本地路径加载 tokenizer 和模型，禁止联网
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        attn_implementation=attn_impl if device.type == "cuda" else None
    ).to(device).eval()
    return model, tokenizer

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
        "\n请将提取结果仅填入以下结构模板中。\n"
        "\n结构模板：\n\n"
        "1. **血糖控制**\n"
        "- 糖化血蛋白：\n"
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
        "3. **其他并发症**\n"
        "- 眼底：\n"
        "- 其他：\n"
        "4. **依从性与监测问题**\n"
        "- 依从性：\n"
        "- 用药情况：\n"
        "5. **生活方式**\n"
        "- BMI：\n"
        "- 饮食：\n"
        "- 运动：\n"
        "- 体重变化：\n\n"
        "\n仅填写以上模板字段，输出到此为止。"
    )
    suffix_ids = tokenizer(structure_suffix, return_tensors=None, add_special_tokens=False)["input_ids"]

    # 给 raw_text 剩下的 token 留下预算
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

def is_valid_output(output, min_sections=5):
    required_sections = [
        "**血糖控制**",
        "**血压管理**",
        "**其他并发症**",
        "**依从性与监测问题**",
        "**生活方式**"
    ]
    matched = [s for s in required_sections if s in output]
    return len(matched), len(matched) >= min_sections

def process_response(text):
    # 移除 <think> 部分
    marker = "</think>"
    pos = text.find(marker)
    if pos != -1:
        end_index = pos + len(marker)
        while end_index < len(text) and text[end_index] == "\n":
            end_index += 1
        text = text[end_index:]

    # 删除所有出现的“结构模版”或“结构模板”
    text = text.replace("结构模版", "").replace("结构模板", "")

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

def query_llm_single(raw_text, max_new_tokens=2200, model=None, tokenizer=None):
    MAX_INPUT_TOKENS = 8000
    MODEL_MAX_LENGTH = 16384

    # 记录输入日志
    logging.info("输入文本：" + raw_text)

    # 兼容旧调用：允许不传模型参数（使用全局变量）
    if model is None:
        model = globals()["model"]
    if tokenizer is None:
        tokenizer = globals()["tokenizer"]

    full_ids = build_prompt_tokens(raw_text, tokenizer, MAX_INPUT_TOKENS)
    total_len = len(full_ids)

    if total_len + max_new_tokens > MODEL_MAX_LENGTH:
        allowed = MODEL_MAX_LENGTH - max_new_tokens
        print(f"[硬性截断] prompt 总 token 数为 {total_len}，超过最大值 {MODEL_MAX_LENGTH - max_new_tokens}，截为前 {allowed} 个 token")
        full_ids = full_ids[:allowed]

    input_ids = torch.tensor(full_ids)
    attention_mask = torch.ones_like(input_ids)

    # 左 padding（这里只处理单条，因此不用 stack）
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
    processed_text = process_response(text)

    # 记录处理结果日志
    logging.info("处理结果：" + processed_text)
    return processed_text

# 如果直接运行该模块，则进入交互模式（仅作测试使用）
if __name__ == "__main__":
    print("开始加载DeepSeek-7B模型...")
    model, tokenizer = load_model_and_tokenizer_7b()
    print("请输入患者病情描述（输入 q 或空行退出）：")
    while True:
        raw = input("\n患者情况：\n")
        if raw.lower() in ['q', 'quit', 'exit'] or raw.strip() == "":
            break
        try:
            result = query_llm_single(raw)
            matched_count, is_valid = is_valid_output(result)
            print("\n模型输出：\n")
            print(result)
            print(f"\n匹配字段数：{matched_count}，状态：{'FULL' if is_valid else 'PARTIAL' if matched_count >= 2 else 'FAIL'}")
        except Exception as e:
            error_msg = f"[错误] 处理失败：{e}"
            print(error_msg)
            logging.error(error_msg)

        torch.cuda.empty_cache()
        gc.collect()

    print("#### 所有任务处理完成 ####")