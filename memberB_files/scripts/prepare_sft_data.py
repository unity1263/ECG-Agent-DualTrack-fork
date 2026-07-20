import json
import os
import random
import re

def heuristic_cot_split(answer, dataset_name):
    """
    启发式将医学回答切分为 <think> 思考过程和最终的 Answer。
    对于医学诊断，通常前 1-2 句是对病理生理、心电图波形或症状的分析，后半部分是诊断结论和治疗建议。
    """
    # 英文断句正则表达式
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    
    # 中文断句支持
    if not sentences or len(sentences) <= 1:
        sentences = re.split(r'(?<=[。！？])\s*', answer.strip())
        
    if len(sentences) > 2:
        # 前两句作为思维链推理过程
        think_part = " ".join(sentences[:2])
        answer_part = " ".join(sentences[2:])
        # 润色 think 模板
        prefix = f"Analyzing clinical presentation and dataset parameters for {dataset_name}. "
        return f"<think>\n{prefix}{think_part}\n</think>\n\n{answer_part}"
    else:
        # 较短的文本，插入标准医学逻辑思考过程
        think_template = (
            f"<think>\n"
            f"1. Evaluate user medical query regarding cardiology/ECG.\n"
            f"2. Reference standard clinical guidelines and pathological criteria.\n"
            f"3. Formulate expert medical advice based on the query: '{answer[:40]}...'\n"
            f"</think>"
        )
        return f"{think_template}\n\n{answer}"

def main():
    corpus_path = r"d:\Users\12869\Desktop\summer\memberA_files\data\cleaned\rag_corpus.json"
    output_dir = r"d:\Users\12869\Desktop\summer\memberB_files\data"
    output_path = os.path.join(output_dir, "sft_train_dataset.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    print(f"Total corpus items: {len(corpus)}")
    
    # 按照数据集分类
    dataset_buckets = {}
    for item in corpus:
        ds = item.get("metadata", {}).get("dataset", "Unknown")
        dataset_buckets.setdefault(ds, []).append(item)
        
    # 定义我们的选取规格
    selection_specs = {
        "Medical inquiry, multiple rounds of dialogue between doctors and patients": 4000,
        "Cross modal diagnosis (ECG Text corresponding diagnosis)": 2500,
        "Long text diagnosis": 1500,
        "Complex diagnosis (analysis of difficult diseases)": 1000,
        "Generate risk assessment": 1000
    }
    
    selected_items = []
    
    # 设置随机种子保证可复现性
    random.seed(42)
    
    # 记录转换统计
    sft_count = 0
    cot_count = 0
    
    for ds_name, target_count in selection_specs.items():
        items = dataset_buckets.get(ds_name, [])
        if not items:
            print(f"Warning: Dataset '{ds_name}' not found or empty.")
            continue
            
        # 如果样本不够，则全量选取
        count_to_select = min(len(items), target_count)
        samples = random.sample(items, count_to_select)
        
        # 确定哪些样本需要 CoT 增强 (主要是 Complex diagnosis 和 Cross modal 中的一部分)
        # Complex diagnosis 全量 CoT，Cross modal 随机 40% CoT
        for idx, item in enumerate(samples):
            content = item.get("content", "")
            # 解析 Question 和 Answer
            q_match = re.search(r"Question:\s*(.*?)\nAnswer:\s*(.*)", content, re.DOTALL)
            if q_match:
                question = q_match.group(1).strip()
                answer = q_match.group(2).strip()
            else:
                # 兼容格式
                question = "Medical inquiry"
                answer = content
                
            # 判断是否应用 CoT
            need_cot = False
            if ds_name == "Complex diagnosis (analysis of difficult diseases)":
                need_cot = True
            elif ds_name == "Cross modal diagnosis (ECG Text corresponding diagnosis)" and idx % 5 < 2: # 40%
                need_cot = True
                
            if need_cot:
                final_answer = heuristic_cot_split(answer, ds_name)
                cot_count += 1
            else:
                final_answer = answer
                sft_count += 1
                
            # 构建 Qwen 聊天格式
            chat_format = {
                "conversations": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert medical AI assistant specializing in cardiology, "
                            "ECG interpretation, and clinical decision support. Provide accurate, "
                            "evidence-based, and structured medical information."
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                ]
            }
            selected_items.append(chat_format)
            
        print(f"Selected {count_to_select} samples from '{ds_name}'")
        
    print(f"Total processed dataset size: {len(selected_items)}")
    print(f"  Standard SFT samples: {sft_count}")
    print(f"  CoT-enhanced samples: {cot_count}")
    
    # 随机打乱整个数据集
    random.shuffle(selected_items)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_items, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
