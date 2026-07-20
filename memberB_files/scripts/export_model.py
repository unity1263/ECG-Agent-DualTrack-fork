import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    lora_dir = r"d:\Users\12869\Desktop\summer\memberB_files\models\deepseek-1.5b-medical-lora"
    export_dir = r"d:\Users\12869\Desktop\summer\memberB_files\models\deepseek-1.5b-medical-merged"
    
    print("=== Model Merge and Export ===")
    print(f"Base Model: {base_model_id}")
    print(f"LoRA Adapter Path: {lora_dir}")
    print(f"Export Target Path: {export_dir}")
    
    if not os.path.exists(lora_dir):
        print(f"[ERROR] LoRA adapter directory does not exist: {lora_dir}")
        return
        
    os.makedirs(export_dir, exist_ok=True)
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    
    print("Loading base model in FP16 (requires ~3.5GB VRAM for merging)...")
    # 合并 LoRA 权重时，必须使用半精度 (Float16/Bfloat16)，不能使用 4/8bit 量化基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",  # 在 CPU 上加载和合并以绝对避免 GPU OOM
        local_files_only=True
    )
    
    print("Loading LoRA adapter and mounting to base model...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        torch_dtype=torch.float16
    )
    
    print("Merging weights (merge_and_unload)...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {export_dir}")
    merged_model.save_pretrained(export_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(export_dir)
    
    print("\n[SUCCESS] Model merged and exported successfully!")

if __name__ == "__main__":
    main()
