# 🏥 ECG-Agent-DualTrack — Medical AI Agent 课程项目

三人合作课程项目：构建面向心电图（ECG）解读与心血管临床决策支持的医疗 AI Agent 系统。

## 项目结构

| 目录 | 负责人 | 内容 |
|------|--------|------|
| `memberA_files/` | Member A | RAG 医疗知识检索库（FAISS + BGE-M3） |
| `memberB_files/` | Member B | DeepSeek-1.5B QLoRA 微调模型 |
| `memberC_files/` | Member C | AI Agent 系统 + RAGAS 评测 + Web Demo |

## Member C 工作概览

### 核心交付物

- **Agent 系统**: 2,770 行 Python 代码
  - `agent/tools.py` — RAG 检索工具 + 模型推理工具
  - `agent/agent_core.py` — 三管道实现 + Agent 编排
  - `agent/build_index.py` — FAISS 索引构建
- **评测框架**: 三管道 RAGAS 对比评测（150条未参练测试集）
- **Web Demo**: Gradio 交互式演示界面
- **终期报告**: `memberC_files/FINAL_REPORT.md`

### 快速开始

```bash
# 安装依赖
pip install -r memberC_files/requirements.txt

# 运行评测
python -m memberC_files.evaluation.run_evaluation --demo

# 启动 Web Demo
python -m memberC_files.demo.app --port 7860
```

### 评测结果摘要

| 管道 | Faithfulness | Answer Relevancy | 速度 |
|------|-------------|-----------------|------|
| Pure RAG (7B + 检索) | 0.5954 | **0.8744** | 中 |
| **Pure SFT (微调 1.5B)** | **0.8673** | 0.8602 | **快** |
| Hybrid Agent (1.5B + RAG) | 0.6179 | 0.8364 | 慢 |

**结论**: QLoRA 微调的小模型在医疗领域表现最优，性价比最高。
