# 🏥 项目资源交接指南 (HANDOVER TO MEMBER C)

本文档旨在为 **成员 C (Agent 架构与量化评测师)** 提供成员 A 与成员 B 已产出资源的完整目录映射、调取接口与后续 Agentic 架构搭建指引。

---

## 一、 项目根目录结构一览

整个项目目录结构清晰规范，包含以下核心模块：

```
summer/
├── HANDOVER_MEMBER_C.md               <-- [本交接指南]
├── AI Agent项目方案生成(1).docx          <-- 课程大纲与题目指导要求
│
├── memberA_files/                     <-- 成员 A 交付 (RAG 向量库与基建)
│   ├── data/cleaned/
│   │   └── rag_corpus.json           # 清洗后的 78,136 条医疗问答语料库 (72MB)
│   ├── models/
│   │   ├── medical_rag_index.faiss   # 构建好的 FAISS 向量检索数据库
│   │   ├── medical_rag_index_docs.pkl # 向量对应的文档块元数据映射
│   │   └── medical_rag_index_embeddings.npy # 高维向量矩阵
│   └── outputs/
│       ├── ablation_report.txt       # RAG 检索超参数消融实验报告
│       └── ablation_visualization.png # 消融实验结果可视化图
│
└── memberB_files/                     <-- 成员 B 交付 (微调模型与适配器)
    ├── data/
    │   └── sft_train_dataset.json     # 10,000 条 Qwen 对话格式微调数据集 (含 2k CoT 样本)
    ├── models/
    │   ├── deepseek-1.5b-medical-merged/ # ★【首选推荐】CPU合并导出的 1.5B 完整模型 (FP16, 3.55GB)
    │   └── deepseek-1.5b-medical-lora/   # 训练产出的 LoRA 权重适配器
    ├── outputs/
    │   ├── loss_curve.png             # 微调训练 Loss 收敛曲线图
    │   ├── before_after_compare.txt   # 5大典型场景微调前后诊断能力对比报告
    │   ├── quantization_report.txt    # 显存压榨与量化策略对比报告
    │   └── vram_profile.txt           # 8GB 显存 Profile 记录 (训练峰值仅 5.59 GB)
    └── scripts/
        ├── prepare_sft_data.py        # 数据抽取与 CoT 启发式切分脚本
        ├── train_qlora.py             # 原生 PyTorch + PEFT 4-bit QLoRA 训练主脚本
        ├── run_inference.py            # 微调前后对比推理评估脚本
        └── export_model.py            # LoRA 权重合并与导出脚本
```

---

## 二、 成员 C 可调用的核心 Python 接口封装

### 1. 调用成员 A 的 RAG 向量检索库

成员 A 经消融实验测试出的最优检索配置为：`chunk_size = 512, chunk_overlap = 50, top_k = 3`。

```python
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class MedicalRAGTool:
    def __init__(self, index_path=r"memberA_files/models/medical_rag_index.faiss",
                       docs_path=r"memberA_files/models/medical_rag_index_docs.pkl"):
        print("Loading FAISS Index and Documents...")
        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
        self.embedder = SentenceTransformer("BAAI/bge-m3")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx]["content"])
        return results

# 测试调用:
# rag_tool = MedicalRAGTool()
# docs = rag_tool.search("房缺 ASD 封堵术后长期预后如何？")
```

### 2. 调用成员 B 的微调医疗模型（DeepSeek-R1-Distill-1.5B）

建议直接使用已合并的文件夹 `memberB_files/models/deepseek-1.5b-medical-merged`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MedicalModelTool:
    def __init__(self, model_path=r"memberB_files/models/deepseek-1.5b-medical-merged"):
        print("Loading Fine-Tuned Medical Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_diagnosis(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert medical AI assistant specializing in cardiology, ECG interpretation, and clinical decision support."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response

# 测试调用:
# model_tool = MedicalModelTool()
# answer = model_tool.generate_diagnosis("患者 ST 段抬高 3mm，胸痛，请诊断。")
```

---

## 三、 成员 C 下一步关键任务（根据大作业要求）

1. **构建简易 AI Agent (推荐 `smolagents`)**：
   - 将上述两个接口封装为 `smolagents` 的 Tools（如 `execute_medical_knowledge_retrieval` 和 `invoke_diagnostic_reasoning`）。
   - 实现 Code-Agent 范式，让小模型直接输出 Python 代码指令路由调用工具。

2. **开展 RAGAS 双轨对比评测**：
   - 提取 100-200 条未参练测试集。
   - 对比三个管道：
     - **管道 1**：纯 RAG (原生 7B 模型 + 检索)
     - **管道 2**：纯 SFT (成员 B 微调后的 1.5B 模型)
     - **管道 3**：混合 Agent (微调 1.5B + RAG 检索)
   - 测量 **Faithfulness (忠实度)** 与 **Answer Relevancy (相关性)**。
   - 尽量给出丰富的评测数据，对比两种方法的优劣点。
3. **尝试在网页端构建出简易的使用demo进行演示**
