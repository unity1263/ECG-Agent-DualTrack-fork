import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """为给定的 prompt 生成回答，并保留 <think> 标签的推理链"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert medical AI assistant specializing in cardiology, "
                "ECG interpretation, and clinical decision support. Provide accurate, "
                "evidence-based, and structured medical information."
            )
        },
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    lora_dir = r"d:\Users\12869\Desktop\summer\memberB_files\models\deepseek-1.5b-medical-lora"
    output_path = r"d:\Users\12869\Desktop\summer\memberB_files\outputs\before_after_compare.txt"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 精选测试集 (5个典型医疗与心肌心电图诊断问题)
    test_prompts = [
        "What are the long-term outcomes of atrial septal defect (ASD) closure in adults?",
        "Patient presents with sudden crushing chest pain, diaphoresis, and dyspnea. ECG shows 3mm ST-segment elevation in leads V1-V4. What is the suspected diagnosis and immediate action plan?",
        "Can a patient request a detailed explanation of the ECG procedure before it begins? What rights do they have?",
        "Explain the significance of QRS changes suggestive of left ventricular hypertrophy (LVH) in a patient with chronic hypertension.",
        "A 65-year-old patient with a history of palpitations has an ECG showing irregular R-R intervals and absent P waves, replaced by rapid fibrillatory waves. What is the diagnosis?"
    ]
    
    # 4-bit 量化加载配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print("Loading base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\n--- [1/2] Loading Base Model (Before Fine-Tuning) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    
    base_responses = []
    for i, prompt in enumerate(test_prompts):
        print(f"Generating base response for Prompt {i+1}...")
        res = generate_response(base_model, tokenizer, prompt)
        base_responses.append(res)
        
    # 释放显存
    del base_model
    torch.cuda.empty_cache()
    
    print("\n--- [2/2] Loading Fine-Tuned Model (With LoRA Adapters) ---")
    # 重新加载基础模型并挂载 LoRA
    from peft import PeftModel
    
    temp_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    
    if not os.path.exists(lora_dir):
        print(f"[WARNING] LoRA weights not found at {lora_dir}. Run training first. Printing base responses only.")
        ft_responses = ["(LoRA not trained yet)"] * len(test_prompts)
    else:
        ft_model = PeftModel.from_pretrained(temp_base_model, lora_dir)
        ft_responses = []
        for i, prompt in enumerate(test_prompts):
            print(f"Generating fine-tuned response for Prompt {i+1}...")
            res = generate_response(ft_model, tokenizer, prompt)
            ft_responses.append(res)
            
        del ft_model
        del temp_base_model
        torch.cuda.empty_cache()
        
    # 保存结果到对比报告
    print(f"\nWriting comparison report to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("        DeepSeek-R1-Distill-Qwen-1.5B 医疗微调前后对比测试报告\n")
        f.write("========================================================================\n\n")
        
        for i, prompt in enumerate(test_prompts):
            f.write(f"测试问题 {i+1}: {prompt}\n")
            f.write("-" * 80 + "\n")
            f.write("【微调前 (Base 1.5B)】:\n")
            f.write(f"{base_responses[i]}\n\n")
            f.write("【微调后 (QLoRA 1.5B)】:\n")
            f.write(f"{ft_responses[i]}\n")
            f.write("=" * 80 + "\n\n")
            
    print("Inference comparison done!")

if __name__ == "__main__":
    main()
