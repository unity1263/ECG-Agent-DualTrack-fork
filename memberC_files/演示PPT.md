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
  或在 VS Code 安装 Marp 插件直接预览
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
6. 成员 C — Agent 架构与评测
7. 评测结果与分析
8. Web Demo 演示
9. 总结与展望

---

## 🎯 项目目标

构建一个面向**心电图解读**与**心血管临床决策支持**的医疗 AI Agent

### 核心技术路线对比

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">

<div style="background:#2e86c1; padding:1.5rem; border-radius:12px;">

### 🔍 RAG 路线
**检索增强生成**

- FAISS 语义检索
- 78K 医疗 QA 语料
- 运行时获取外部知识
- "大模型 + 知识库"

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
│🔴 RAG │ │🔵 SFT│ │🟢HBD │
│7B模型 │ │1.5B  │ │1.5B  │
│+ 检索 │ │微调  │ │+ RAG │
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
| **模型** | 外部 7B | 微调 1.5B | 微调 1.5B |
| **知识** | FAISS 检索 | 模型权重 | 权重 + 检索 |
| **思路** | 规模换质量 | 微调换质量 | 取长补短 |
| **显存** | ~5 GB | ~2 GB | ~3 GB |
| **速度** | 中 | ⚡快 | 慢 |

---

## 📦 成员 A — RAG 知识检索库

### 数据与索引

- **语料规模**: 78,136 条清洗后医疗 QA 对（6 大医学领域）
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
✅ 低成本训练  ✅ 快速推理  ✅ 医学领域知识内化

---

## 🤖 成员 C — Agent 架构与评测

### 核心交付

- **Agent 系统** (~2,700 行 Python)
  - `tools.py` — RAG检索工具 + 模型推理工具
  - `agent_core.py` — 三管道实现 + smolagents CodeAgent
  - `build_index.py` — FAISS索引构建

- **双轨评测框架**
  - `test_dataset.py` — 150条未参练测试集（分层抽样）
  - `run_evaluation.py` — RAGAS 指标计算
  - `analyze_results.py` — 可视化图表生成

- **Gradio Web Demo** — 交互式中文问答界面

---

## 📊 评测结果

### RAGAS 指标对比

| 管道 | Faithfulness | Answer Relevancy | 速度 |
|------|:----------:|:-------------:|:---:|
| 🔴 Pure RAG | 0.5954 | **0.8744** | 中 |
| 🔵 **Pure SFT** | **0.8673** | 0.8602 | **快** |
| 🟢 Hybrid | 0.6179 | 0.8364 | 慢 |

### 关键发现

> **微调后的小模型在忠实度上显著优于大模型+RAG方案**
>
> 将医学知识"内化"到 1.5B 参数中，比运行时检索更可靠

---

## 🌐 Web Demo

![width:800px](https://via.placeholder.com/800x450/1a5276/ffffff?text=Medical+AI+Agent+Web+Demo)

### 功能特点
- 🖥 **简洁中文界面** — Gradio 交互式 Web UI
- 🚀 **GPU 真机推理** — DeepSeek-1.5B + LoRA 实时生成
- 📚 **RAG 增强** — FAISS 检索 78K 医疗文献
- 🔄 **三模式切换** — 混合智能体 / 纯SFT / 纯RAG

---

## 💡 结论与展望

### 核心结论

1. **小模型微调 > 大模型检索**
   - QLoRA 微调的 1.5B 模型在医疗领域 Faithfulness 最优 (0.867)
   - 知识"内化"比"检索"更可靠

2. **QLoRA 极低显存训练**
   - 1.5B 模型仅需 5.6GB，8GB 消费卡即可训练

3. **Agent 架构有提升空间**
   - smolagents CodeAgent 的检索-推理协调可进一步优化

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
- 🌐 Demo: `http://localhost:7860`

**Made with ❤️ for AI Agent Course**
