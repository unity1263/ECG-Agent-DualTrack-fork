---
marp: true
theme: uncover
class:
  - lead
  - invert
paginate: true
size: 16:9
---

<!--
  医疗 AI Agent 课程项目 PPT
  使用 Marp 渲染: npx @marp-team/marp-cli slides.md -o slides.pdf
-->

# 🏥 医疗 AI Agent 系统

## 心电图解读 & 心血管临床决策支持

**AI Agent 课程大作业**

Member A · Member B · Member C

---

## 📋 目录

1. 项目背景与目标
2. 系统架构总览
3. 三管道评测设计
4. 成员 A — RAG 知识检索库
5. 成员 B — 医学模型微调
6. 成员 C — Agent 架构
7. 评测方法论 ★
8. 评测结果与分析
9. Web Demo 演示
10. 总结与展望

---

## 🎯 项目目标

构建面向**心电图解读**与**心血管临床决策支持**的医疗 AI Agent

### 核心技术路线对比

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">

<div style="background:#2e86c1; padding:1.5rem; border-radius:12px;">

### 🔍 RAG 路线
**检索增强生成**

- FAISS 语义检索
- 78K 医疗 QA 语料
- 运行时获取外部知识
- "模型 + 知识库"

</div>

<div style="background:#27ae60; padding:1.5rem; border-radius:12px;">

### 🧠 SFT 路线
**监督微调**

- DeepSeek-1.5B
- 10K 医疗 QA 训练
- 知识内化到模型参数
- "小模型 + 领域知识"

</div>

</div>

---

## 🏗 系统架构

```
用户提问 (Web / CLI)
    │
    ▼
┌──────────────────────────┐
│   🎯 MedicalAgentRunner  │  ◄── 统一调度入口
└──┬────────┬────────┬─────┘
   │        │        │
   ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐
│🔴RAG │ │🔵SFT │ │🟢HBD │
│1.5B  │ │1.5B  │ │1.5B  │
│+检索 │ │微调  │ │+RAG  │
└──┬───┘ └──┬───┘ └──┬───┘
   │        │        │
   ▼        ▼        ▼
┌──────────────────────────┐
│     📊 RAGAS 评测框架     │
│ Faithfulness + Relevancy  │
└──────────────────────────┘
```

---

## 🔬 三管道对比设计

|  | 🔴 管道1 | 🔵 管道2 | 🟢 管道3 |
|--|:--:|:--:|:--:|
| **名称** | Pure RAG | Pure SFT | Hybrid Agent |
| **模型** | 1.5B base | 1.5B QLoRA | 1.5B QLoRA |
| **知识** | FAISS 检索 | 模型权重 | 权重 + 检索 |
| **Agent** | 固定工作流 | 直接推理 | CodeAgent 编排 |
| **显存** | ~4 GB | ~2 GB | ~4 GB |
| **速度** | 中 | ⚡快 | 慢 |

---

## 📦 成员 A — RAG 知识检索库

### 数据与索引

- **语料规模**: 78,136 条清洗后医疗 QA 对（6 大医学领域）——消融实验 F1=0.8628
- **嵌入模型**: BAAI/bge-m3（1024 维多语言向量）
- **索引算法**: FAISS IVF-Flat（近似最近邻搜索）
- **最优参数**（消融实验）:
  - chunk_size = 512, overlap = 50
  - top_k = 3，F1 = **0.8628**

### 涵盖领域
基础心脏病学 · 跨模态ECG诊断 · 复杂疾病分析 · 风险评估 · 临床对话

---

## 🧪 成员 B — 医学模型 QLoRA 微调

### 训练配置

| 参数 | 值 |
|------|-----|
| 基座模型 | DeepSeek-R1-Distill-Qwen-**1.5B** |
| 微调方法 | **4-bit QLoRA** (NF4, Double Quant) |
| 训练数据 | 10,000 条医学 QA（含 2,000 CoT） |
| 训练峰值显存 | **仅 5.59 GB**（8GB 消费卡即可训练） |
| LoRA 权重 | ~71 MB |

### 优势
✅ 低成本训练 ✅ 快速推理 ✅ 医学领域知识内化

---

## 🤖 成员 C — Agent 架构

### 核心交付

- **Agent 系统** (~2,700 行 Python)
  - `tools.py` — RAG检索 + 模型推理两大工具（懒加载，smolagents 兼容）
  - `agent_core.py` — 三管道实现 + CodeAgent 编排 + rule-based fallback
  - `build_index.py` — FAISS索引构建（FP16加速）

- **评测框架**
  - `test_dataset.py` — MD5去重 + 5领域分层抽样 → 150条公平测试集
  - `run_evaluation.py` — 三管道并发推理 + RAGAS 指标计算
  - `analyze_results.py` — 4种可视化图表

- **Gradio Web Demo** — 中文交互界面 + GPU真机推理

---

## 📊 评测方法论 ★

### 如何评估 Agent 效果？

**1. 测试集准备（保证公平）**
- 78K 语料 → MD5 哈希去重（排除 SFT 训练集 10K 条）
- 5 领域分层抽样 → **150 条未参练测试题**
- 每领域 20%，seed=42 保证可复现

**2. RAGAS 双指标**
- **Faithfulness（忠实度）**: 答案能否从上下文推断？→ 防止编造
- **Answer Relevancy（相关性）**: 答案是否切题？→ 防止跑题

**3. 评测流程**
- 同一问题 → 三管道并行推理 → 统一 RAGAS 打分
- 记录耗时，支持 GPU 真机 / CPU demo 双模式

---

## 📊 评测结果对比

### RAGAS 指标（150 条测试集，启发式评分）

| 管道 | Faithfulness | Answer Relevancy | 速度 |
|------|:----------:|:-------------:|:---:|
| 🔴 Pure RAG (1.5B + 检索) | 0.5888 | **0.8914** | 0.50s |
| 🔵 **Pure SFT** (微调 1.5B) | **0.8644** | 0.8812 | **0.15s** |
| 🟢 Hybrid Agent (1.5B + RAG) | 0.6120 | 0.8422 | 0.80s |

### 关键发现

> **QLoRA 微调在 Faithfulness 上完胜 RAG（+46.8%）**
>
> 知识"内化"到参数中比运行时检索更可靠。
> Hybrid Agent 未达预期，Agent 编排引入了不确定性。

---

## 🌐 Web Demo

### 功能特点
- 🖥 **简洁中文界面** — Gradio 交互式 Web UI
- 🚀 **GPU 真机推理** — DeepSeek-1.5B + LoRA + RTX 3060
- 📚 **RAG 增强** — FAISS 检索 78K 医疗文献
- 🔄 **三模式切换** — 混合智能体 / 纯SFT / 纯RAG
- 📊 **评测数据展示** — RAGAS 指标内嵌页面

---

## 💡 总结与展望

### 核心结论

1. **小模型微调 > 大模型检索**
   - QLoRA 微调的 1.5B 模型在 Faithfulness 上最优 (0.867)
   - 知识"内化"比"检索"更可靠

2. **QLoRA 极低显存训练**
   - 1.5B 模型仅需 5.6GB，8GB 消费卡即可训练

3. **评测方法论可复用**
   - 去重→分层抽样→三管道→RAGAS→可视化
   - 适用于任何垂直领域 AI Agent 评测

### 未来方向
- 🔬 更大规模 CoT 微调数据
- 🧩 多模态（ECG 图像 + 文本）
- 🏥 真实临床场景验证

---

## 🙏 谢谢！

### 项目资源

- 📂 GitHub: `ECG-Agent-DualTrack`
- 📄 架构文档: `ARCHITECTURE.md`
- 📝 终期报告: `memberC_files/FINAL_REPORT.md`
- 🎤 演讲稿: `memberC_files/演讲稿.md`
- 🌐 Demo: `http://localhost:7860`

**Made with ❤️ for AI Agent Course**
