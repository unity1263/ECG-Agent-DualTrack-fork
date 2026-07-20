import os
import sys

# 设置环境变量，使用国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

def main():
    base_dir = r"d:\Users\12869\Desktop\summer\memberB_files\base_models"
    model_dir = os.path.join(base_dir, "deepseek-r1-1.5b")
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Target directory: {model_dir}")
    print("Connecting to Hugging Face Mirror (hf-mirror.com)...")
    
    try:
        snapshot_download(
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            local_dir=model_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pt"],
            resume_download=True
        )
        print("\n[SUCCESS] Model downloaded successfully!")
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
