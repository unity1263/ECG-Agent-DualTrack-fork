"""
医疗 AI Agent — 交互式 Web 演示 (Gradio).

提供一个简洁的中文问答界面，支持三种推理管道。
"""

import os
import sys
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- 检查依赖 ---
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

# --- 预置示例回答 (无 GPU 时使用) ---
DEMO_ANSWERS = {
    "pure_rag": {
        "answer": "根据检索到的医学文献和临床指南，该患者的临床表现符合**急性前壁ST段抬高型心肌梗死（STEMI）**。\n\n"
                  "**诊断依据：**\n"
                  "1. V1-V4导联ST段抬高≥3mm，提示前壁透壁性心肌缺血\n"
                  "2. 胸痛、大汗、呼吸困难为典型急性心梗三联征\n"
                  "3. 左前降支（LAD）是可能的犯罪血管\n\n"
                  "**紧急处理措施：**\n"
                  "1. 立即行12导联心电图确认\n"
                  "2. 监测心肌标志物（hs-cTnI）\n"
                  "3. 双联抗血小板治疗（阿司匹林+P2Y12抑制剂）\n"
                  "4. 急诊PCI，目标门-球时间≤90分钟",
        "time": 0.5,
    },
    "pure_sft": {
        "answer": "**诊断：急性前壁ST段抬高型心肌梗死（Anterior STEMI）**\n\n"
                  "**处理方案：**\n"
                  "- 立即启动急诊PCI导管室\n"
                  "- 双联抗血小板治疗\n"
                  "- 肝素抗凝\n"
                  "- 持续心电监护",
        "time": 0.15,
    },
    "hybrid_agent": {
        "answer": "**综合分析（智能体模式）**\n\n"
                  "**诊断：** 急性前壁ST段抬高型心肌梗死\n\n"
                  "**推理过程：**\n"
                  "- ST段在V1-V4抬高3mm → 前壁透壁性缺血 → LAD闭塞可能性大\n"
                  "- 胸痛+大汗+呼吸困难为典型ACS表现\n\n"
                  "**处理建议（依据ACC/AHA指南）：**\n"
                  "1. MONA方案：吗啡、吸氧、硝酸甘油、阿司匹林\n"
                  "2. 启动导管室 → 急诊PCI\n"
                  "3. 双联抗血小板 + 抗凝治疗\n"
                  "4. 持续监护",
        "time": 0.8,
    },
}

EXAMPLES = [
    ["患者突发胸痛、大汗、呼吸困难，心电图显示V1-V4导联ST段抬高3mm。最可能的诊断是什么？应该采取什么紧急措施？"],
    ["65岁患者，有房颤病史，心电图显示R-R间期绝对不齐，P波消失代之以大小不等的f波。请给出诊断。"],
    ["房间隔缺损（ASD）封堵术后，成人患者的长期预后如何？"],
    ["糖尿病合并高血压患者，55岁，心电图V3-V6导联ST段压低，肌钙蛋白I升高。诊断与处理？"],
]


# ============================================================================
# 后端
# ============================================================================
class DemoBackend:
    """处理问答请求，有 GPU 时走真实推理，无 GPU 用预生成回答。"""

    def __init__(self):
        self._available = False
        self._runner = None
        self._try_load()

    def _try_load(self):
        try:
            import torch
            if torch.cuda.is_available():
                from memberC_files.agent.agent_core import MedicalAgentRunner, PipelineMode
                self._runner = MedicalAgentRunner()
                self._PipelineMode = PipelineMode
                self._available = True
                logger.info("✅ GPU 可用，启用真实推理。")
            else:
                logger.info("⚠️ 未检测到 GPU，使用演示模式。")
        except Exception as e:
            logger.info(f"⚠️ ML 依赖不可用 ({e})，使用演示模式。")

    def ask(self, question: str, mode: str) -> tuple:
        """处理用户问题，返回 (回答, 管道信息, 耗时)。"""
        if not question or not question.strip():
            return "⚠️ 请输入医学问题。", "", ""

        if self._available:
            try:
                # Map demo mode names to PipelineMode enum values
                mode_map = {"pure_rag": "pure_rag", "pure_sft": "pure_sft", "hybrid_agent": "hybrid"}
                pipe = self._PipelineMode(mode_map.get(mode, mode))
                import time
                t0 = time.time()
                result = self._runner.run(question, pipe)
                t = time.time() - t0
                answer = result.get("answer", "未能生成回答。")
                info = f"管道: {result.get('pipeline', mode)}"
                time_str = f"⏱ {t:.1f}s"
                return answer, info, time_str
            except Exception as e:
                logger.error(f"推理失败: {e}")
                # 出错时回退到演示模式
                pass

        # 演示模式
        import time
        r = DEMO_ANSWERS.get(mode, DEMO_ANSWERS["hybrid_agent"])
        answer = r["answer"]
        mode_names = {"pure_rag": "纯 RAG", "pure_sft": "纯 SFT", "hybrid_agent": "混合智能体"}
        info = f"管道: {mode_names.get(mode, mode)} (演示模式)"
        time_str = f"⏱ {r['time']:.2f}s (模拟)"
        time.sleep(0.3)
        return answer, info, time_str


# ============================================================================
# UI
# ============================================================================
def build_ui(backend: DemoBackend) -> gr.Blocks:
    with gr.Blocks(title="🏥 医疗AI智能体 — 心电图与心血管临床决策支持") as demo:

        # --- 标题 ---
        gr.Markdown("""
        # 🏥 医疗 AI 智能体
        ### 心电图解读 & 心血管临床决策支持
        """)

        # --- 输入区 ---
        question_input = gr.Textbox(
            label="📝 请输入医学问题 / 临床场景",
            placeholder="例如：患者突发胸痛、大汗、呼吸困难，心电图V1-V4导联ST段抬高3mm。最可能的诊断是什么？",
            lines=3,
        )

        # --- 模式选择 ---
        with gr.Row():
            mode_radio = gr.Radio(
                choices=[
                    ("🟢 混合智能体 (1.5B + RAG 检索)", "hybrid_agent"),
                    ("🔵 纯 SFT (微调 1.5B 模型)", "pure_sft"),
                    ("🔴 纯 RAG (检索 + 大模型)", "pure_rag"),
                ],
                value="hybrid_agent",
                label="推理模式",
            )
            submit_btn = gr.Button("🔍 开始分析", variant="primary", size="lg")

        # --- 状态 ---
        with gr.Row():
            mode_label = gr.Textbox(label="当前管道", value="混合智能体", interactive=False, scale=1)
            time_label = gr.Textbox(label="耗时", value="", interactive=False, scale=1)

        # --- 输出 ---
        answer_output = gr.Markdown(
            value="*请在上方输入医学问题，点击「开始分析」获取 AI 回答。*",
            label="分析结果",
        )

        # --- 示例 ---
        gr.Markdown("### 💡 试试这些问题")
        gr.Examples(examples=EXAMPLES, inputs=[question_input])

        # --- 评测数据 ---
        with gr.Accordion("📊 三管道评测数据 (RAGAS)", open=True):
            gr.Markdown("""
            ### 🔬 RAGAS 双轨对比评测结果

            评测基于 **150 条未参练医学问答**（5 大领域分层抽样），使用 RAGAS 框架计算 Faithfulness（忠实度）和 Answer Relevancy（答案相关性）。

            | 管道 | 模型 | Faithfulness | Answer Relevancy | 推理速度 | 显存 |
            |------|------|:----------:|:-------------:|:------:|:---:|
            | 🔴 **Pure RAG** | DeepSeek-1.5B + FAISS检索 | 0.5954 | **0.8744** | 中 | ~3 GB |
            | 🔵 **Pure SFT** | DeepSeek-1.5B QLoRA | **0.8673** | 0.8602 | **快** ⚡ | ~2 GB |
            | 🟢 **Hybrid Agent** | 1.5B QLoRA + RAG | 0.6179 | 0.8364 | 慢 | ~4 GB |

            ### 💡 关键结论

            > **QLoRA 微调在忠实度上最优，知识"内化"比"检索"更可靠。**
            >
            > Pure SFT 的 Faithfulness 达 **0.8673**，远超 Pure RAG 的 0.5954（+45.6%）。
            > Hybrid Agent 的 Faithfulness (0.6179) 介于两者之间，说明 Agent 编排带来了折中效果。

            ### 📋 评测数据集

            | 领域 | 描述 | 占比 |
            |------|------|:--:|
            | 基础心脏病学 | Cardiology Knowledge QA | 20% |
            | 跨模态 ECG 诊断 | ECG Text Diagnosis | 20% |
            | 复杂疾病分析 | Complex Diagnosis | 20% |
            | 风险评估 | Risk Assessment | 20% |
            | 临床对话 | Clinical Dialogue | 20% |

            *测试集从 78,136 条语料中排除 SFT 训练集后分层抽样（seed=42），确保评测公平性。*
            """)

        # --- 说明 ---
        with gr.Accordion("📖 关于三种模式", open=False):
            gr.Markdown("""
            | 模式 | 模型 | 知识来源 | 特点 |
            |------|------|---------|------|
            | 🟢 **混合智能体** | DeepSeek-1.5B + LoRA | 模型权重 + RAG检索 | 取长补短，推荐使用 |
            | 🔵 **纯 SFT** | DeepSeek-1.5B + LoRA | 仅模型权重 | 速度快，知识内化模型 |
| 🔴 **纯 RAG** | DeepSeek-1.5B (基础) | FAISS 检索 | 基础模型+外部知识库 |
            """)

        gr.Markdown("---\n*⚠️ 本项目为课程作业，仅供教学演示，不可用于临床诊断。*")

        # --- 事件绑定 ---
        def on_submit(question, mode):
            answer, info, time_str = backend.ask(question, mode)
            return answer, info, time_str

        submit_btn.click(
            fn=on_submit,
            inputs=[question_input, mode_radio],
            outputs=[answer_output, mode_label, time_label],
        )
        question_input.submit(
            fn=on_submit,
            inputs=[question_input, mode_radio],
            outputs=[answer_output, mode_label, time_label],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="医疗 AI Agent Web 演示")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not HAS_GRADIO:
        print("错误：需要安装 Gradio。请运行: pip install gradio")
        sys.exit(1)

    print("=" * 50)
    print("  🏥 医疗 AI 智能体 — Web Demo")
    print("=" * 50)
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  模式: {'GPU 真实推理' if torch.cuda.is_available() else '演示模式'}")
    print("=" * 50)

    backend = DemoBackend()
    demo = build_ui(backend)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    import torch
    main()
