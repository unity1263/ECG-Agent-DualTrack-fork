import os
import sys
import json
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

class MedicalSFTDataset(Dataset):
    """自定义 PyTorch 医疗数据集类，免除对 datasets 库的依赖"""
    def __init__(self, json_path, tokenizer, max_len=512, limit=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        if limit is not None:
            self.data = self.data[:limit]
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item["conversations"]
        
        # 使用模型的对话模版进行渲染
        text = self.tokenizer.apply_chat_template(conversations, tokenize=False)
        
        # Tokenize 文本
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,  # 动态 padding 延迟到 collate_fn
            return_tensors=None
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.copy()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch, pad_token_id=0):
    """动态 padding 数据整理函数，只填充到当前 batch 的最大长度以节省显存与时间"""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for ids, mask, label in zip(input_ids, attention_mask, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_token_id] * pad_len)
        padded_attention_mask.append(mask + [0] * pad_len)
        # 在计算因果语言模型 Loss 时，padding 部分的 label 设置为 -100 以在交叉熵中被忽略
        padded_labels.append(label + [-100] * pad_len)
        
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long)
    }

def plot_loss_curve(losses, output_path):
    """绘制并保存 Loss 收敛曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.title('DeepSeek-R1-Distill-Qwen-1.5B QLoRA Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Step (× logging_steps)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve successfully saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical LLM QLoRA Fine-tuning")
    parser.add_argument("--smoke_test", action="store_true", help="Run in smoke test mode (10 samples, 5 steps)")
    args = parser.parse_args()
    
    # 路径配置
    model_dir = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    dataset_path = r"d:\Users\12869\Desktop\summer\memberB_files\data\sft_train_dataset.json"
    output_dir = r"d:\Users\12869\Desktop\summer\memberB_files\models\deepseek-1.5b-medical-lora"
    plot_path = r"d:\Users\12869\Desktop\summer\memberB_files\outputs\loss_curve.png"
    vram_log_path = r"d:\Users\12869\Desktop\summer\memberB_files\outputs\vram_profile.txt"
    
    if args.smoke_test:
        output_dir = r"d:\Users\12869\Desktop\summer\memberB_files\models\smoke_test_output"
        print("=== RUNNING IN SMOKE TEST MODE ===")
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    # GPU 状态检测
    print("=== [1/5] GPU Configuration ===")
    if not torch.cuda.is_available():
        print("CUDA GPU is required for fine-tuning.")
        sys.exit(1)
    device = torch.device("cuda")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {vram_total:.2f} GB")
    
    print("=== [2/5] Quantizing and Loading Base Model ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    # k-bit 训练准备
    model = prepare_model_for_kbit_training(model)
    vram_after_load = torch.cuda.memory_allocated(0) / 1e9
    print(f"Base model loaded. VRAM usage: {vram_after_load:.2f} GB")
    
    print("=== [3/5] Configuring LoRA Adapters ===")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 显式激活模型梯度检查点
    model.gradient_checkpointing_enable()
    
    print("=== [4/5] Preparing Dataset & Dataloader ===")
    # 限制或加载完整样本
    sample_limit = 10 if args.smoke_test else None
    train_dataset = MedicalSFTDataset(dataset_path, tokenizer, max_len=512, limit=sample_limit)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )
    print(f"Dataset Size: {len(train_dataset)}, DataLoader Batches: {len(train_loader)}")
    
    print("=== [5/5] Starting Native PyTorch Training Loop ===")
    # QLoRA 最优 AdamW 8-bit 优化器
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4)
    
    # Cosine 学习率衰减模拟
    import math
    num_epochs = 1 if args.smoke_test else 3
    gradient_accumulation_steps = 1 if args.smoke_test else 8
    total_steps = len(train_loader) * num_epochs
    max_train_steps = 5 if args.smoke_test else total_steps
    
    print(f"Total training epochs: {num_epochs}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Max train steps: {max_train_steps}")
    
    scaler = torch.amp.GradScaler('cuda') # 混合精度 GradScaler
    
    step_count = 0
    losses = []
    log_losses = []
    
    torch.cuda.reset_peak_memory_stats(0)
    model.train()
    
    try:
        for epoch in range(num_epochs):
            if step_count >= max_train_steps:
                break
                
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            optimizer.zero_grad()
            
            for batch_step, batch in enumerate(train_loader):
                if step_count >= max_train_steps:
                    break
                    
                # tensors 迁入 GPU
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # FP16 混合精度前向计算
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                # 归一化 loss
                loss = loss / gradient_accumulation_steps
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积更新
                if (batch_step + 1) % gradient_accumulation_steps == 0 or (batch_step + 1) == len(train_loader):
                    # 梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    step_count += 1
                    current_loss = loss.item() * gradient_accumulation_steps
                    losses.append(current_loss)
                    
                    if step_count % 1 == 0:  # 每步打印
                        print(f"Step {step_count}/{max_train_steps} - Loss: {current_loss:.4f}")
                        
        print("\nTraining Finished!")
        vram_peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"Peak VRAM during training: {vram_peak:.2f} GB")
        
        # 写入显存占用日志
        if not args.smoke_test:
            with open(vram_log_path, "w", encoding="utf-8") as f:
                f.write("NVIDIA RTX 4060 (8GB VRAM) Fine-Tuning Profile:\n")
                f.write("---------------------------------------------\n")
                f.write(f"Base model size (1.5B Qlora loaded): {vram_after_load:.2f} GB\n")
                f.write(f"Peak VRAM during training:           {vram_peak:.2f} GB\n")
                f.write(f"Memory overhead of activations & grad: {vram_peak - vram_after_load:.2f} GB\n")
            print(f"VRAM Profile logged to: {vram_log_path}")
            
            # 保存 LoRA 权重
            print(f"Saving LoRA weights to: {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # 绘制并保存 Loss 收敛图
            plot_loss_curve(losses, plot_path)
        else:
            print("\n[SUCCESS] Smoke test completed successfully!")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
