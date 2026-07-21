"""
Medical AI Agent — Interactive Web Demo (Gradio).

Provides a user-friendly web interface for the Medical AI Agent system.
Users can:
  - Input medical questions / clinical scenarios
  - Select pipeline mode (Pure RAG, Pure SFT, or Hybrid Agent)
  - View generated answers with retrieval context
  - Compare results across all three pipelines side-by-side
  - See example questions for quick testing

Usage:
    python -m memberC_files.demo.app
    python -m memberC_files.demo.app --port 7860 --share
"""
import os
import sys
import json
import time
import logging
import argparse
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config.settings import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Check if Gradio is available
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    logger.error(
        "Gradio is required for the web demo. Install with: pip install gradio"
    )


# ============================================================================
# Demo Mode — generates sample responses when ML deps are not available
# ============================================================================
DEMO_RESPONSES = {
    "pure_rag": {
        "answer": (
            "Based on the retrieved medical literature and clinical guidelines, "
            "the patient presentation is consistent with acute coronary syndrome. "
            "Immediate management should include:\n\n"
            "1. **Initial Assessment**: 12-lead ECG within 10 minutes of arrival\n"
            "2. **Cardiac Biomarkers**: High-sensitivity troponin measurement\n"
            "3. **Immediate Therapy**: \n"
            "   - Aspirin 162-325 mg chewed\n"
            "   - Nitroglycerin for ongoing chest pain\n"
            "   - Oxygen if SpO2 < 90%\n"
            "4. **Reperfusion Strategy**: Primary PCI within 90 minutes if STEMI confirmed\n\n"
            "The ST-segment elevation in V1-V4 indicates anteroseptal MI, likely due to "
            "LAD occlusion. Early reperfusion is critical for myocardial salvage."
        ),
        "contexts": [
            "ST-segment elevation myocardial infarction (STEMI) is diagnosed when there is "
            "ST-segment elevation ≥1 mm in two contiguous limb leads or ≥2 mm in two contiguous "
            "precordial leads. Anteroseptal MI (V1-V4) typically results from occlusion of the "
            "left anterior descending (LAD) artery. Primary PCI is the preferred reperfusion "
            "strategy when available within 90-120 minutes of first medical contact."
        ],
        "time": 0.45,
    },
    "pure_sft": {
        "answer": (
            "The clinical presentation indicates acute ST-elevation myocardial infarction "
            "(STEMI) of the anteroseptal wall. Immediate action plan:\n"
            "- Activate emergency cardiac catheterization team\n"
            "- Administer dual antiplatelet therapy (aspirin + P2Y12 inhibitor)\n"
            "- Consider anticoagulation with heparin\n"
            "- Urgent coronary angiography with intent for primary PCI"
        ),
        "contexts": [],
        "time": 0.15,
    },
    "hybrid_agent": {
        "answer": (
            "**Clinical Analysis (Hybrid Agent)**\n\n"
            "After retrieving relevant medical knowledge and performing diagnostic reasoning:\n\n"
            "**Diagnosis**: Acute Anteroseptal ST-Elevation Myocardial Infarction (STEMI)\n\n"
            "**Rationale**:\n"
            "- ST elevation of 3mm in V1-V4 indicates transmural ischemia in the anteroseptal region\n"
            "- The LAD artery is the likely culprit vessel\n"
            "- Sudden crushing chest pain with diaphoresis is classic for acute MI\n\n"
            "**Immediate Actions** (per ACC/AHA Guidelines):\n"
            "1. MONA protocol: Morphine, Oxygen, Nitroglycerin, Aspirin\n"
            "2. Activate cath lab for primary PCI (goal: door-to-balloon ≤ 90 min)\n"
            "3. Start dual antiplatelet therapy\n"
            "4. Continuous cardiac monitoring\n\n"
            "**Evidence Support**: Retrieved clinical guidelines confirm this management approach "
            "with Class I recommendation for primary PCI in STEMI."
        ),
        "contexts": [
            "ACC/AHA STEMI Guidelines: For patients with STEMI presenting within 12 hours "
            "of symptom onset, primary PCI is recommended (Class I, Level of Evidence A). "
            "Door-to-balloon time should be ≤ 90 minutes.",
            "ECG Interpretation: ST elevation in leads V1-V4 characteristically indicates "
            "anteroseptal myocardial infarction due to LAD occlusion proximal to the first "
            "septal perforator."
        ],
        "time": 0.80,
    },
}


# ============================================================================
# Predefined example questions
# ============================================================================
EXAMPLE_QUESTIONS = [
    [
        "Patient presents with sudden crushing chest pain, diaphoresis, and dyspnea. "
        "ECG shows 3mm ST-segment elevation in leads V1-V4. What is the suspected "
        "diagnosis and immediate action plan?"
    ],
    [
        "A 65-year-old patient with a history of palpitations has an ECG showing "
        "irregular R-R intervals and absent P waves, replaced by rapid fibrillatory "
        "waves. What is the diagnosis?"
    ],
    [
        "What are the long-term outcomes of atrial septal defect (ASD) closure in adults?"
    ],
    [
        "Explain the significance of QRS changes suggestive of left ventricular "
        "hypertrophy (LVH) in a patient with chronic hypertension."
    ],
    [
        "A 55-year-old diabetic patient presents with nausea, fatigue, and mild chest "
        "discomfort. ECG shows ST depression in leads V3-V6. Troponin I is elevated. "
        "What is the diagnosis and management?"
    ],
    [
        "What is the Wells criteria for pulmonary embolism and how is it applied "
        "in the emergency department setting?"
    ],
]


# ============================================================================
# Core demo logic
# ============================================================================
class DemoBackend:
    """
    Backend for the web demo. Handles question processing with the selected pipeline.

    Falls back to pre-generated demo responses when ML dependencies are not installed,
    allowing the UI to be tested and demonstrated without GPU access.
    """

    def __init__(self):
        self._pipelines_available = False
        self._runner = None
        self._try_load_pipelines()

    def _try_load_pipelines(self):
        """Try to load actual ML pipelines; fall back to demo mode."""
        try:
            import torch
            if torch.cuda.is_available():
                from agent.agent_core import MedicalAgentRunner, PipelineMode
                self._runner = MedicalAgentRunner()
                self._PipelineMode = PipelineMode
                self._pipelines_available = True
                logger.info("GPU available — using real model inference.")
            else:
                logger.info("No GPU — using demo mode.")
        except ImportError as e:
            logger.info(f"ML dependencies not available ({e}) — using demo mode.")

    def process_question(
        self,
        question: str,
        mode: str = "hybrid",
        progress: Optional[gr.Progress] = None,
    ) -> tuple:
        """
        Process a medical question through the selected pipeline.

        Args:
            question: Medical question text
            mode: "pure_rag" | "pure_sft" | "hybrid" | "compare_all"
            progress: Gradio progress tracker

        Returns:
            Tuple of (answer_text, contexts_text, pipeline_info, time_text)
        """
        if not question or not question.strip():
            return (
                "⚠️ Please enter a medical question.",
                "",
                "No question provided",
                "",
            )

        if self._pipelines_available and mode != "compare_all":
            # Real inference
            pipe_mode = self._PipelineMode(mode)
            result = self._runner.run(question, pipe_mode)

            answer = result.get("answer", "No answer generated.")
            contexts = "\n\n---\n\n".join(
                result.get("retrieved_docs", []) or result.get("contexts", [])
            ) or "(No retrieval context — direct model inference)"
            info = f"Pipeline: {result.get('pipeline', 'unknown')}"
            time_str = f"Response generated"

        elif mode == "compare_all":
            # Side-by-side comparison
            if self._pipelines_available:
                from agent.agent_core import PipelineMode
                results = self._runner.run_all_pipelines(question)

                answer_parts = []
                for pipe_name, pipe_label in [
                    ("pure_rag", "🔴 Pipeline 1: Pure RAG (7B + Retrieval)"),
                    ("pure_sft", "🔵 Pipeline 2: Pure SFT (Fine-tuned 1.5B)"),
                    ("hybrid", "🟢 Pipeline 3: Hybrid Agent (1.5B + RAG)"),
                ]:
                    r = results.get(pipe_name, {})
                    answer_parts.append(f"### {pipe_label}\n{r.get('answer', 'N/A')}\n")

                answer = "\n---\n\n".join(answer_parts)
                contexts = "(Comparison mode — each pipeline may use different context)"
                info = "All three pipelines compared"
                time_str = "Comparison complete"
            else:
                # Demo comparison
                answer_parts = []
                for pipe_key, pipe_label in [
                    ("pure_rag", "🔴 Pipeline 1: Pure RAG (7B + Retrieval)"),
                    ("pure_sft", "🔵 Pipeline 2: Pure SFT (Fine-tuned 1.5B)"),
                    ("hybrid", "🟢 Pipeline 3: Hybrid Agent (1.5B + RAG)"),
                ]:
                    r = DEMO_RESPONSES[pipe_key]
                    answer_parts.append(f"### {pipe_label}\n{r['answer']}\n")

                answer = "\n---\n\n".join(answer_parts)
                contexts = "### 🔴 Pure RAG Context\n" + DEMO_RESPONSES["pure_rag"]["contexts"][0] \
                    + "\n\n### 🟢 Hybrid Agent Context\n" + DEMO_RESPONSES["hybrid_agent"]["contexts"][0]
                info = "All three pipelines (demo mode)"
                time_str = (
                    f"Pure RAG: {DEMO_RESPONSES['pure_rag']['time']:.2f}s | "
                    f"Pure SFT: {DEMO_RESPONSES['pure_sft']['time']:.2f}s | "
                    f"Hybrid: {DEMO_RESPONSES['hybrid_agent']['time']:.2f}s"
                )

        else:
            # Single pipeline demo mode
            r = DEMO_RESPONSES.get(mode, DEMO_RESPONSES["hybrid"])
            answer = r["answer"]
            if r["contexts"]:
                contexts = "\n\n---\n\n".join(
                    f"**Document {i+1}**: {c}"
                    for i, c in enumerate(r["contexts"])
                )
            else:
                contexts = "(No retrieval context — direct model inference)"
            info = f"Pipeline: {mode} (demo mode)"
            time_str = f"Response time: {r['time']:.2f}s"

        return answer, contexts, info, time_str


# ============================================================================
# Build Gradio UI
# ============================================================================
def build_ui(backend: DemoBackend) -> gr.Blocks:
    """Build the Gradio web interface."""

    # Custom CSS for medical-themed styling
    custom_css = """
    .medical-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .medical-header h1 {
        font-size: 2em;
        margin: 0;
    }
    .medical-header p {
        font-size: 1.1em;
        opacity: 0.9;
    }
    .pipeline-rag { border-left: 4px solid #E63946; }
    .pipeline-sft { border-left: 4px solid #457B9D; }
    .pipeline-hybrid { border-left: 4px solid #2A9D8F; }
    footer { visibility: hidden; }
    """

    with gr.Blocks(
        css=custom_css,
        title="Medical AI Agent | ECG & Cardiology",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
    ) as demo:

        # Header
        gr.HTML("""
        <div class="medical-header">
            <h1>🏥 Medical AI Agent</h1>
            <p>ECG Interpretation & Cardiology Clinical Decision Support</p>
            <p style="font-size: 0.85em; opacity: 0.8;">
                Powered by DeepSeek-R1-Distill-Qwen-1.5B + RAG Knowledge Retrieval
            </p>
        </div>
        """)

        # Main layout: sidebar + content
        with gr.Row():
            # Sidebar — controls
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### ⚙️ Configuration")

                pipeline_mode = gr.Radio(
                    choices=[
                        ("🟢 Hybrid Agent (1.5B + RAG)", "hybrid"),
                        ("🔴 Pure RAG (7B + Retrieval)", "pure_rag"),
                        ("🔵 Pure SFT (Fine-tuned 1.5B)", "pure_sft"),
                        ("📊 Compare All Three", "compare_all"),
                    ],
                    value="hybrid",
                    label="Pipeline Mode",
                    info="Choose how the agent processes your question",
                )

                gr.Markdown("---")
                gr.Markdown("### 💡 Example Questions")
                examples = gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=[gr.Textbox(label="Question", visible=False)],
                    label="Click an example to try",
                )

                gr.Markdown("---")
                gr.Markdown("""
                ### 📋 Pipeline Info

                | Pipeline | Model | Knowledge |
                |----------|-------|-----------|
                | **Hybrid Agent** | 1.5B Fine-tuned | RAG + SFT |
                | **Pure RAG** | 7B Base | Retrieval |
                | **Pure SFT** | 1.5B Fine-tuned | Weights only |
                """)

                submit_btn = gr.Button(
                    "🔍 Analyze Medical Question",
                    variant="primary",
                    size="lg",
                )

            # Main content — results
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="📝 Medical Question / Clinical Scenario",
                    placeholder=(
                        "Describe the patient's symptoms, ECG findings, and clinical "
                        "context. E.g.: 'Patient presents with crushing chest pain, "
                        "ST elevation 3mm in V1-V4...'"
                    ),
                    lines=4,
                )

                with gr.Row():
                    pipeline_badge = gr.Textbox(
                        label="Active Pipeline",
                        value="Hybrid Agent (demo mode)",
                        interactive=False,
                        scale=1,
                    )
                    time_display = gr.Textbox(
                        label="Response Time",
                        value="",
                        interactive=False,
                        scale=1,
                    )

                with gr.Tabs():
                    with gr.TabItem("📋 Generated Answer", id="answer_tab"):
                        answer_output = gr.Markdown(
                            value="*Enter a medical question and click 'Analyze' to begin.*",
                            label="Answer",
                            elem_classes=["pipeline-hybrid"],
                        )

                    with gr.TabItem("📚 Retrieved Context", id="context_tab"):
                        context_output = gr.Markdown(
                            value="*Retrieved medical knowledge will appear here.*",
                            label="Retrieved Context",
                        )

                # Evaluation metrics (shown in compare mode)
                with gr.Accordion("📊 Evaluation Metrics", open=False):
                    gr.Markdown("""
                    Metrics are computed when running in **Compare All** mode:
                    - **Faithfulness**: How well the answer is grounded in retrieved context
                    - **Answer Relevancy**: How relevant the answer is to the question
                    - **Response Time**: Wall-clock time for answer generation
                    """)

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 15px; color: #666; font-size: 0.85em;">
            <p>🩺 Medical AI Agent — Course Project | Data from 78K cleaned medical QA pairs
            | Model: DeepSeek-R1-Distill-Qwen-1.5B QLoRA fine-tuned</p>
            <p><strong>⚠️ Disclaimer</strong>: This is an educational project. Not for clinical use.</p>
        </div>
        """)

        # Wire up interactions
        def on_submit(question, mode):
            answer, contexts, info, time_str = backend.process_question(question, mode)
            return answer, contexts, info, time_str

        submit_btn.click(
            fn=on_submit,
            inputs=[question_input, pipeline_mode],
            outputs=[answer_output, context_output, pipeline_badge, time_display],
        )

        # Enable Enter key to submit
        question_input.submit(
            fn=on_submit,
            inputs=[question_input, pipeline_mode],
            outputs=[answer_output, context_output, pipeline_badge, time_display],
        )

    return demo


# ============================================================================
# Launch
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Medical AI Agent Web Demo")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    if not HAS_GRADIO:
        print("=" * 60)
        print("ERROR: Gradio is required for the web demo.")
        print("Please install dependencies:")
        print("  pip install -r memberC_files/requirements.txt")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("  Medical AI Agent — Web Demo")
    print("=" * 60)
    print(f"  Starting server on http://{args.host}:{args.port}")
    print()
    print("  Pipeline modes:")
    print("    🟢 Hybrid Agent  — Fine-tuned 1.5B + RAG (recommended)")
    print("    🔴 Pure RAG       — 7B base model + retrieval")
    print("    🔵 Pure SFT       — Fine-tuned 1.5B only")
    print("    📊 Compare All     — Side-by-side comparison")
    print("=" * 60)

    backend = DemoBackend()
    demo = build_ui(backend)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
