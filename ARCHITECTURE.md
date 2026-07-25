# 🏥 ECG-Agent-DualTrack — 项目架构设计文档

**版本**: v1.0  
**日期**: 2026年7月24日  
**课程**: AI Agent 大作业  
**小组成员**: Member A (数据与RAG检索库) · Member B (模型微调) · Member C (Agent架构与评测)

---

## 目录

1. [项目总览](#1-项目总览)
2. [整体架构](#2-整体架构)
3. [模块设计](#3-模块设计)
4. [数据流](#4-数据流)
5. [技术栈](#5-技术栈)
6. [部署与运行](#6-部署与运行)
7. [评测体系](#7-评测体系)

---

## 1. 项目总览

### 1.1 目标

构建面向**心电图（ECG）解读**与**心血管临床决策支持**的医疗 AI Agent 系统，对比"检索增强生成 (RAG)"与"监督微调 (SFT)"两条技术路线的优劣。

### 1.2 核心交付

| 成员 | 模块 | 核心产出 |
|------|------|---------|
| A | RAG 知识检索库 | FAISS 向量索引 + 78K 医疗 QA 语料 |
| B | QLoRA 微调模型 | DeepSeek-1.5B + LoRA 适配器 |
| C | Agent 系统 + 评测 + Demo | 三管道 Agent、RAGAS 评测框架、Gradio 交互演示 |

---

## 2. 整体架构

### 2.1 系统架构图

```
                                    ┌──────────────────────────────┐
                                    │     👤 用户 (Web / CLI)       │
                                    └─────────────┬────────────────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          │                       │                       │
                          ▼                       ▼                       ▼
              ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
              │  📋 Gradio Web UI │ │  🖥️ CLI Interface  │ │  📊 Evaluation Runner │
              │  (demo/app.py)    │ │  (agent_core.py)   │ │  (run_evaluation.py)  │
              └─────────┬─────────┘ └─────────┬─────────┘ └───────────┬───────────┘
                        │                     │                       │
                        └─────────────────────┼───────────────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  MedicalAgentRunner │ ◄── 统一调度入口
                                    │  (agent_core.py)    │
                                    └─┬──────┬──────┬────┘
                                      │      │      │
                   ┌──────────────────┘      │      └──────────────────┐
                   ▼                         ▼                         ▼
    ┌──────────────────────────┐ ┌─────────────────────────┐ ┌──────────────────────────┐
    │  🔴 Pipeline 1           │ │  🔵 Pipeline 2          │ │  🟢 Pipeline 3            │
    │  Pure RAG                │ │  Pure SFT               │ │  Hybrid Agent             │
    │  (7B + Retrieval)        │ │  (Fine-tuned 1.5B)      │ │  (1.5B + RAG)             │
    └───────────┬──────────────┘ └───────────┬─────────────┘ └───────────┬──────────────┘
                │                            │                            │
      ┌─────────▼─────────┐        ┌─────────▼─────────┐        ┌─────────▼─────────┐
      │  RAG Tool         │        │  Model Tool        │        │  CodeAgent (smol) │
      │  (semantic search) │        │  (direct generate) │        │  chooses RAG/Model │
      └─────────┬─────────┘        └─────────┬─────────┘        └─────────┬─────────┘
                │                            │                            │
      ┌─────────▼──────────────────────────┐ │  ┌─────────────────────────▼──────┐
      │  FAISS Index (IVF-Flat)            │ │  │  LoRA Adaptors                  │
      │  ├─ 78K docs, BAAI/bge-m3         │ │  │  ├─ DeepSeek-R1-Distill-Qwen   │
      │  ├─ dim=1024, nlist=273            │ │  │  ├─ 1.5B params, 4-bit NF4     │
      │  └─ cosine similarity (IP norm'd)  │ │  │  └─ 10K medical QA fine-tuned  │
      └────────────────────────────────────┘ │  └────────────────────────────────┘
                                             │
      ┌──────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────┐
│  评测框架 (RAGAS)                                      │
│  ├─ Faithfulness (忠实度)     — 答案是否基于上下文？      │
│  ├─ Answer Relevancy (相关性)  — 答案是否切题？          │
│  ├─ Context Precision/Recall  — 检索质量评估            │
│  └─ Time/Quality Trade-off    — 速度与质量权衡          │
└──────────────────────────────────────────────────────┘
```

### 2.2 三层架构

```
┌────────────────────────────────────────────────────────────┐
│  表示层 (Presentation Layer)                                 │
│  ├─ Gradio Web UI        — 交互式医疗问答演示                │
│  ├─ CLI Interface        — 命令行问答入口                    │
│  └─ Visualization        — matplotlib 图表生成              │
├────────────────────────────────────────────────────────────┤
│  业务逻辑层 (Business Logic Layer)                           │
│  ├─ MedicalAgentRunner   — 三管道调度与生命周期管理           │
│  ├─ PureRAGPipeline      — RAG 检索 + 大模型生成             │
│  ├─ PureSFTPipeline      — 微调模型直接推理                  │
│  ├─ HybridAgentPipeline  — CodeAgent 编排 RAG + 模型        │
│  └─ RAGASEvaluator       — 启发式/库级评测指标计算            │
├────────────────────────────────────────────────────────────┤
│  工具层 (Tools Layer)                                        │
│  ├─ MedicalRAGTool       — FAISS 语义检索引擎               │
│  └─ MedicalModelTool     — 微调医疗模型推理引擎              │
├────────────────────────────────────────────────────────────┤
│  基础设施层 (Infrastructure Layer)                           │
│  ├─ FAISS IVF-Flat Index — 向量相似度检索（148K+ 向量）       │
│  ├─ BAAI/bge-m3          — 1024维文本嵌入模型                │
│  ├─ DeepSeek-1.5B        — QLoRA 4-bit 微调基座模型          │
│  ├─ LoRA Adapters        — PEFT 低秩适配权重                 │
│  └─ 78K Medical QA Corpus — 六大医学领域 QA 语料             │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 模块设计

### 3.1 模块依赖关系

```
memberC_files/
│
├── config/settings.py          ◄── 全局配置中心（所有模块依赖）
│   └── 路径、模型名、超参数、量化配置
│
├── agent/                      ◄── Agent 核心
│   ├── tools.py                ◄── 工具层（RAG 检索 + 模型推理）
│   │   ├── MedicalRAGTool      ──→ FAISS Index + bge-m3 + pickle(docs)
│   │   └── MedicalModelTool    ──→ DeepSeek-1.5B + LoRA + BitsAndBytes
│   ├── agent_core.py           ◄── 业务逻辑层（三管道 + 调度器）
│   │   ├── PureRAGPipeline     ──→ MedicalRAGTool + 7B External Model
│   │   ├── PureSFTPipeline     ──→ MedicalModelTool (direct)
│   │   ├── HybridAgentPipeline ──→ MedicalRAGTool + MedicalModelTool + smolagents
│   │   └── MedicalAgentRunner  ──→ 统一入口
│   └── build_index.py          ◄── 一次性索引构建脚本
│       └── rag_corpus.json → chunk → embed → FAISS index
│
├── evaluation/                 ◄── 评测框架
│   ├── test_dataset.py         ──→ 从 78K 语料中提取 150 条未参练测试集
│   ├── run_evaluation.py       ──→ 三管道并发评测 + RAGAS 指标计算
│   │   └── RAGASEvaluator      ──→ heuristic / ragas library
│   └── analyze_results.py      ──→ 可视化（柱状图/雷达图/散点图）
│
├── demo/app.py                 ◄── Gradio 交互式 Web Demo
│   ├── DemoBackend             ──→ MedicalAgentRunner (GPU) / pre-generated (CPU)
│   └── Gradio UI               ──→ 3 pipeline modes + compare all
│
└── outputs/ / data/            ◄── 输出产物
    ├── eval_test_dataset.json  (150 条测试集)
    ├── evaluation_detailed.json (逐题明细)
    ├── evaluation_summary.json  (汇总指标)
    └── evaluation_comparison.csv (CSV 对比表)
```

### 3.2 核心类设计

#### 3.2.1 工具层

| 类名 | 文件 | 职责 | 关键方法 |
|------|------|------|---------|
| `MedicalRAGTool` | `agent/tools.py` | FAISS 语义检索 + BAAI/bge-m3 嵌入 | `search(query) → list[dict]`, `forward(query) → str` |
| `MedicalModelTool` | `agent/tools.py` | DeepSeek-1.5B QLoRA 推理 | `generate(prompt) → str`, `forward(prompt) → str` |

**设计要点**:
- **懒加载机制**: 两个工具均使用 `_ensure_loaded()` 延迟加载，首次调用时才初始化模型/索引，避免不必要的内存/显存占用
- **smolagents 兼容**: 均实现 `forward()` 方法和 `name`/`description` 属性，可直接作为 smolagents Tool 注册
- **工厂模式**: `create_rag_tool()` / `create_model_tool()` 提供单例式创建入口

**MedicalRAGTool 检索流程**:
```
query → embed(bge-m3, 1024d) → L2 normalize → FAISS.search(IP, top_k=3)
       → lookup docs[metadata] → return [{content, score, source}]
```

**MedicalModelTool 推理流程**:
```
prompt → chat_template(system + user messages)
       → tokenize → model.generate(max_new_tokens=512, temp=0.6, top_p=0.9)
       → decode → return response
```

#### 3.2.2 业务层 — 三管道

| 类名 | 管道 | 模型 | 知识来源 | Agent 范式 |
|------|------|------|---------|-----------|
| `PureRAGPipeline` | 🔴 Pipeline 1 | Qwen2-7B-Instruct (外部) | FAISS Top-3 检索 | 无 Agent — 固定 workflow |
| `PureSFTPipeline` | 🔵 Pipeline 2 | DeepSeek-1.5B + LoRA | 模型权重（隐式） | 无 Agent — 直接推理 |
| `HybridAgentPipeline` | 🟢 Pipeline 3 | DeepSeek-1.5B + LoRA | RAG + 权重 | smolagents CodeAgent / rule-based |

**Pipeline 1 — Pure RAG 工作流**:
```
Question → RAG.search(question) → format context → augmented_prompt → 7B model generate → answer
```

**Pipeline 2 — Pure SFT 工作流**:
```
Question → chat_template → fine-tuned 1.5B model generate → answer
```

**Pipeline 3 — Hybrid Agent 工作流**:
```
Question → CodeAgent.plan()
               ├─ 需要知识 → execute_medical_knowledge_retrieval(question) → context
               ├─ 需要推理 → invoke_diagnostic_reasoning(prompt + context) → analysis
               └─ 综合 → final answer
```

**Agent 双重策略**:
1. **smolagents CodeAgent**（首选）: Agent 自主决策，输出 Python 代码调用工具
2. **Rule-based Fallback**（备用）: Always RAG → Augment Prompt → Model Generate

#### 3.2.3 调度层 — MedicalAgentRunner

```
MedicalAgentRunner
├─ _pipelines: dict[PipelineMode, Pipeline]  # Lazy instantiation
├─ run(question, mode) → dict                # 单管道执行
└─ run_all_pipelines(question) → dict        # 三管道并发比较
```

**统一返回格式**:
```python
{
    "question": str,
    "answer": str,
    "pipeline": "pure_rag" | "pure_sft" | "hybrid_agent",
    "retrieved_docs": list[str],    # RAG/Hybrid 有
    "retrieval_scores": list[float], # RAG/Hybrid 有
    "agent_type": "smolagents" | "rule_based"  # Hybrid 有
}
```

#### 3.2.4 评测层

| 类名 | 文件 | 职责 |
|------|------|------|
| `EvalSample` | `evaluation/run_evaluation.py` | 单个评测样本的数据结构（dataclass） |
| `RAGASEvaluator` | `evaluation/run_evaluation.py` | RAGAS 指标计算（库级 / 启发式 fallback） |
| `EvaluationRunner` | `evaluation/run_evaluation.py` | 评测编排器：加载数据 → 跑三管道 → 算指标 → 保存 |

**RAGAS 指标计算策略**:

| 指标 | 有 ragas 库时 | 无 ragas 库时（启发式） |
|------|-------------|----------------------|
| **Faithfulness** | LLM 判断答案是否可从句子的上下文推断 |  答案词与上下文词的重叠率 × 1.2 |
| **Answer Relevancy** | LLM 基于问题-答案相似度 | 关键词重叠(0.4) + 长度得分(0.3) + 医学术语奖励(0.3) |
| **Context Precision** | 检索文档与问题的相关性 | 未实现 |
| **Context Recall** | 所有相关文档是否被检索到 | 未实现 |

**启发式计算的局限性与设计意图**:
- 启发式指标用于"评测框架的可运行性验证"——在没有 GPU/完整 ML 依赖时也能跑通流程
- 生产环境应使用 RAGAS 库的 LLM-based 指标
- 代码已预留 `compute_ragas_batch()` 方法，安装 ragas 即可切换

#### 3.2.5 演示层

| 类名 | 文件 | 职责 |
|------|------|------|
| `DemoBackend` | `demo/app.py` | 处理问答请求，有 GPU 时走真实推理，无 GPU 时用预生成响应 |

**展示模式**:
- **Hybrid Agent** — 混合智能体（默认推荐）
- **Pure RAG** — 纯检索增强
- **Pure SFT** — 纯微调模型
- **Compare All** — 三管道并行对比（并排展示）

---

## 4. 数据流

### 4.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│  阶段 0: 数据准备 (Member A & B)                                         │
│                                                                          │
│  6 Medical Datasets ──► 清洗去重 ──► 78,136 QA pairs (rag_corpus.json)   │
│                                      │                                   │
│                                      ├──► chunk(512w, overlap=50)        │
│                                      │    └──► embed(bge-m3, 1024d)      │
│                                      │         └──► FAISS IVF-Flat Index │
│                                      │              └──► memberA/models/ │
│                                      │                                   │
│                                      └──► 抽样 10K ──► SFT format        │
│                                           └──► QLoRA train               │
│                                                └──► LoRA adapters         │
│                                                     └──► memberB/models/ │
├─────────────────────────────────────────────────────────────────────────┤
│  阶段 1: 测试集提取 (Member C)                                            │
│                                                                          │
│  rag_corpus.json ──► 哈希去重(排除SFT训练集) ──► 分层抽样(5领域)           │
│                    └──► 150 samples ──► eval_test_dataset.json            │
├─────────────────────────────────────────────────────────────────────────┤
│  阶段 2: 三管道评测 (Member C)                                            │
│                                                                          │
│  eval_test_dataset.json                                                  │
│      │                                                                   │
│      ├──► Pipeline 1: question → RAG.search → 7B.generate → answer1      │
│      ├──► Pipeline 2: question → 1.5B.generate → answer2                 │
│      └──► Pipeline 3: question → Agent(RAG+Model) → answer3              │
│      │                                                                   │
│      └──► RAGAS: faithfulness + answer_relevancy + time                  │
│           └──► evaluation_detailed.json + evaluation_summary.json        │
│                └──► matplotlib → bar/radar/scatter charts (.png)         │
├─────────────────────────────────────────────────────────────────────────┤
│  阶段 3: 交互式演示 (Member C)                                            │
│                                                                          │
│  User Input → Gradio UI → DemoBackend                                    │
│      ├── GPU available → MedicalAgentRunner → real inference             │
│      └── No GPU → DEMO_RESPONSES (预生成) → demo mode response           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 关键数据文件

| 文件 | 大小 | 来源 | 内容 |
|------|------|------|------|
| `memberA_files/data/cleaned/rag_corpus.json` | 72MB | Member A | 78,136 条清洗后医疗 QA |
| `memberA_files/models/medical_rag_index.faiss` | ~290MB | Member A/build_index | FAISS IVF-Flat 向量索引 |
| `memberB_files/data/sft_train_dataset.json` | — | Member B | 10,000 条 SFT 训练数据 |
| `memberB_files/models/deepseek-1.5b-medical-lora/` | — | Member B | LoRA 适配器权重 |
| `memberC_files/data/eval_test_dataset.json` | — | Member C | 150 条未参练测试集（5领域分层） |
| `memberC_files/outputs/evaluation_detailed.json` | — | Member C | 逐题三管道评测明细 |

### 4.3 评测数据集分布

测试集从 78K 语料中排除 SFT 训练集的 10K 样本后提取，按 5 大医学领域分层抽样：

| 领域 | 描述 | 占比 |
|------|------|------|
| `cardiology_basic` | 心脏病学基础知识 | ~20% |
| `cross_modal_ecg` | 跨模态 ECG 诊断 | ~20% |
| `complex_diagnosis` | 复杂病诊断分析 | ~20% |
| `risk_assessment` | 风险评估生成 | ~20% |
| `clinical_dialogue` | 医患多轮对话 | ~20% |

---

## 5. 技术栈

### 5.1 核心技术组件

| 层级 | 技术 | 版本要求 | 说明 |
|------|------|---------|------|
| **嵌入模型** | BAAI/bge-m3 | — | 1024 维多语言嵌入，支持中英混合 |
| **向量检索** | FAISS (IVF-Flat) | ≥1.7.4 | CPU 版，nlist=273，内积相似度 |
| **基座模型** | DeepSeek-R1-Distill-Qwen-1.5B | — | 1.5B 参数，用于 SFT + Agent |
| **外部模型** | Qwen2-7B-Instruct | — | 7B 参数，用于 Pure RAG 管道 |
| **微调方法** | PEFT LoRA | ≥0.8.0 | 低秩适配，4-bit NF4 量化 |
| **量化框架** | BitsAndBytes | ≥0.41.0 | load_in_4bit, nf4, double_quant |
| **Agent 框架** | smolagents | ≥1.0.0 | CodeAgent 范式，Python 代码路由 |
| **评测框架** | RAGAS | ≥0.1.0 | faithfulness, answer_relevancy 等 |
| **Web 框架** | Gradio | ≥4.0.0 (实测 6.20) | 交互式医疗问答 UI |
| **可视化** | matplotlib + numpy | — | 柱状图、雷达图、散点图 |

### 5.2 量化配置

```python
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,           # 4-bit 量化加载
    "bnb_4bit_quant_type": "nf4",   # NormalFloat4 量化类型
    "bnb_4bit_use_double_quant": True, # 双重量化（节省更多显存）
    "bnb_4bit_compute_dtype": "float16", # 计算精度保持 FP16
}
```

### 5.3 FAISS 索引参数

| 参数 | 值 | 设计依据 |
|------|---|---------|
| chunk_size | 512 words | Member A 消融实验最优值 (F1=0.8628) |
| chunk_overlap | 50 words | 平衡上下文连续性与冗余 |
| top_k | 3 | 检索精度与上下文长度权衡 |
| 索引算法 | IVF-Flat (IP) | 内积相似度 + L2 归一化 = 余弦相似度 |
| nlist | 273 | ≈ sqrt(78K), 平衡速度与精度 |
| 嵌入维度 | 1024 | BAAI/bge-m3 输出维度 |

---

## 6. 部署与运行

### 6.1 环境要求

```
Python: 3.10+
GPU: 推荐 (至少 8GB VRAM 用于 1.5B 4-bit 模型)
内存: 16GB+ (加载 FAISS 索引 + 模型)
```

### 6.2 依赖安装

```bash
pip install -r memberC_files/requirements.txt
```

### 6.3 启动命令

```bash
# 1. 构建 FAISS 索引（如果 memberA 已提供可跳过）
python -m memberC_files.agent.build_index

# 2. 提取评测测试集（如果不存在）
python -m memberC_files.evaluation.test_dataset

# 3. 运行三管道评测（demo mode: 无 GPU 也能跑通框架）
python -m memberC_files.evaluation.run_evaluation --demo

# 4. 生成评测可视化图表
python -m memberC_files.evaluation.analyze_results

# 5. 启动 Web Demo
python -m memberC_files.demo.app --port 7860 --host 0.0.0.0

# 6. CLI 交互式问答
python -m memberC_files.agent.agent_core -q "患者 ST 段抬高 3mm，胸痛，请诊断。" -m hybrid
```

### 6.4 双模式运行

项目设计了**真机推理**和**演示模式**两种运行方式：

| 运行模式 | 条件 | 推理 | 首选项 |
|----------|------|------|-------|
| **真机模式** | CUDA GPU + torch 可用 | 加载模型进行真实推理 | `--demo false` |
| **演示模式** （默认） | 无 GPU / 依赖缺失 | 使用预生成的 DEMO_RESPONSES | `--demo` (default) |

演示模式的目的是：在不具备 GPU 环境时也能完整跑通**评测框架逻辑**和**Web UI 交互流程**。

---

## 7. 评测体系

### 7.1 双轨对比设计

本项目的核心实验设计是对比两条技术路线的效果：

| 维度 | 检索增强 (RAG) | 监督微调 (SFT) |
|------|--------------|--------------|
| **代表管道** | Pipeline 1: 7B + RAG | Pipeline 2: Fine-tuned 1.5B |
| **核心理念** | "用更大的模型 + 外部知识" | "用小模型 + 领域知识" |
| **模型规模** | 7B 参数 | 1.5B 参数 |
| **知识获取** | 运行时检索 | 训练时注入 |
| **显存需求** | ~14 GB (7B 量化) | ~4 GB (1.5B 量化) |
| **推理延迟** | ~0.5s 检索 + 生成 | ~0.15s 纯生成 |

### 7.2 评测指标

| 指标 | 英文名 | 计算方式 | 意义 |
|------|--------|---------|------|
| **忠实度** | Faithfulness | 答案是否可被上下文支撑 | 衡量 RAG 答案的事实准确性 |
| **答案相关性** | Answer Relevancy | 答案与问题的语义匹配度 | 衡量模型是否在回答正确的问题 |
| **检索精度** | Context Precision | 检索文档与问题的相关性 | 衡量 RAG 检索质量 |
| **检索召回** | Context Recall | 所有相关文档是否被检索到 | 衡量检索覆盖度 |

### 7.3 评测结果摘要（来自终期报告）

| 管道 | Faithfulness | Answer Relevancy | 推理速度 | 显存 |
|------|-------------|-----------------|---------|------|
| **Pure RAG** (7B + 检索) | 0.5954 | **0.8744** | 中 | 高 |
| **Pure SFT** (微调 1.5B) | **0.8673** | 0.8602 | **快** | **低** |
| **Hybrid Agent** (1.5B + RAG) | 0.6179 | 0.8364 | 慢 | 中 |

**关键结论**: QLoRA 微调的小模型在医疗领域 Faithfulness 最高（0.8673），说明将医学知识"内化"到模型参数中比运行时检索更有效。Hybrid Agent 的 Faithfulness 较低可能因 Agent 的检索-推理协调引入了更多不确定性。

---

## 附：文件清单

```
ECG-Agent-DualTrack-fork/
│
├── README.md                         # 项目说明与快速开始
├── ARCHITECTURE.md                   # [本文档] 架构设计文档
├── HANDOVER_MEMBER_C.md              # 成员 A/B → C 的交接指南
│
├── memberA_files/                    # Member A — RAG 知识检索库
│   ├── data/cleaned/
│   │   ├── rag_corpus.json           # 78,136 条清洗后医疗 QA (72MB)
│   │   └── *.png / *.txt             # 数据清洗质量报告
│   ├── models/
│   │   ├── medical_rag_index.faiss   # FAISS IVF-Flat 向量索引
│   │   ├── medical_rag_index_docs.pkl # 文档元数据映射
│   │   └── medical_rag_index_embeddings.npy # 高维嵌入矩阵
│   └── outputs/
│       ├── ablation_results.csv/json # 消融实验结果
│       └── ablation_visualization.png # 消融可视化
│
├── memberB_files/                    # Member B — QLoRA 微调模型
│   ├── data/
│   │   └── sft_train_dataset.json    # 10K SFT 训练数据
│   ├── models/
│   │   ├── deepseek-1.5b-medical-lora/ # LoRA 适配器权重
│   │   └── deepseek-1.5b-medical-merged/ # (可选) 合并后完整模型
│   ├── scripts/
│   │   ├── prepare_sft_data.py       # 数据预处理
│   │   ├── train_qlora.py            # QLoRA 训练脚本
│   │   ├── run_inference.py          # 微调前后对比推理
│   │   └── export_model.py           # LoRA 合并导出
│   └── outputs/
│       ├── loss_curve.png            # 训练 Loss 曲线
│       ├── before_after_compare.txt  # 微调前后能力对比
│       ├── quantization_report.txt   # 量化策略报告
│       └── vram_profile.txt          # 8GB 显存 Profile
│
└── memberC_files/                    # Member C — Agent + 评测 + Demo
    ├── config/
    │   └── settings.py               # 全局配置中心
    ├── agent/
    │   ├── tools.py                  # MedicalRAGTool + MedicalModelTool
    │   ├── agent_core.py             # 三管道 + MedicalAgentRunner
    │   └── build_index.py            # FAISS 索引构建
    ├── evaluation/
    │   ├── test_dataset.py           # 测试集提取
    │   ├── run_evaluation.py         # 评测运行器
    │   └── analyze_results.py        # 可视化分析
    ├── demo/
    │   └── app.py                    # Gradio Web UI
    ├── data/
    │   └── eval_test_dataset.json    # 150 条评测测试集
    ├── outputs/
    │   ├── evaluation_detailed.json  # 逐题评测明细
    │   ├── evaluation_summary.json   # 汇总指标
    │   └── evaluation_comparison.csv # CSV 对比表
    ├── FINAL_REPORT.md               # 终期报告
    ├── 实验报告.md                   # 实验详细报告
    └── requirements.txt              # Python 依赖
```

---

> **撰写人**: Claude (based on source code analysis)  
> **最后更新**: 2026年7月24日
