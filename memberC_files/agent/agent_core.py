"""
Medical AI Agent — Code-Agent Orchestration Core.

This module implements the Medical AI Agent using the smolagents Code-Agent paradigm.
The agent can autonomously decide when to:
  - Retrieve medical knowledge (RAG)
  - Invoke diagnostic reasoning (Fine-tuned LLM)
  - Combine both for hybrid answers

Architecture (3 pipelines for evaluation comparison):
  Pipeline 1 — Pure RAG:     External 7B model + RAG retrieval context
  Pipeline 2 — Pure SFT:     Fine-tuned 1.5B model (no retrieval)
  Pipeline 3 — Hybrid Agent: Fine-tuned 1.5B + RAG retrieval (Agent-driven)
"""
import os
import sys
import json
import logging
from typing import Optional
from enum import Enum

import torch

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from memberC_files.config.settings import (
    OUTPUT_DIR,
    AGENT_SYSTEM_PROMPT,
    INFERENCE_MAX_NEW_TOKENS,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P,
    BASE_MODEL_7B,
    BASE_MODEL_1_5B,
    LORA_MODEL_PATH,
    QUANTIZATION_CONFIG,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Enum
# ============================================================================
class PipelineMode(Enum):
    """Evaluation pipeline modes for the dual-track comparison."""
    PURE_RAG = "pure_rag"       # 7B base model + RAG context
    PURE_SFT = "pure_sft"       # Fine-tuned 1.5B model only
    HYBRID_AGENT = "hybrid"     # Fine-tuned 1.5B + RAG (Agent-driven)


# ============================================================================
# Pipeline 1: Pure RAG (7B Model + Knowledge Retrieval)
# ============================================================================
class PureRAGPipeline:
    """
    Pipeline 1: External large model (7B) + RAG retrieval.

    Workflow:
      1. User asks a medical question
      2. RAG retrieves top-k relevant medical documents
      3. Documents are formatted as context
      4. 7B model generates answer using question + retrieved context

    This represents the "scale via retrieval" approach — using a larger model
    with external knowledge instead of fine-tuning a smaller one.
    """

    def __init__(self, base_model_id: str = BASE_MODEL_7B):
        self.base_model_id = base_model_id
        self._model = None
        self._tokenizer = None
        self._rag_tool = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading Pipeline 1 (Pure RAG) — Base model: {self.base_model_id}")

        bnb_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, trust_remote_code=True
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self._model.eval()

        # Load RAG
        from memberC_files.agent.tools import MedicalRAGTool
        self._rag_tool = MedicalRAGTool()

        self._loaded = True
        logger.info("Pipeline 1 (Pure RAG) loaded.")

    def answer(self, question: str, max_new_tokens: int = INFERENCE_MAX_NEW_TOKENS) -> dict:
        """
        Answer a medical question using RAG-augmented 7B model.

        Returns dict with keys: question, retrieved_docs, answer, pipeline
        """
        self._ensure_loaded()

        # Step 1: Retrieve relevant knowledge
        retrieved = self._rag_tool.search(question)

        # Step 2: Build augmented prompt
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            context_parts.append(f"[Document {i}] (source: {r['source']})\n{r['content']}")
        context = "\n\n".join(context_parts)

        augmented_prompt = f"""You are a medical AI assistant. Use the following reference documents to answer the question.
If the documents are not relevant, use your own medical knowledge.

Reference Documents:
{context}

Question: {question}

Based on the reference documents and your medical knowledge, provide a comprehensive answer:"""

        # Step 3: Generate
        messages = [
            {"role": "system", "content": "You are an expert medical AI assistant."},
            {"role": "user", "content": augmented_prompt},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=INFERENCE_TEMPERATURE,
                top_p=INFERENCE_TOP_P,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        return {
            "question": question,
            "retrieved_docs": [r["content"] for r in retrieved],
            "retrieval_scores": [r["score"] for r in retrieved],
            "answer": response.strip(),
            "pipeline": "pure_rag",
        }


# ============================================================================
# Pipeline 2: Pure SFT (Fine-Tuned 1.5B Model Only)
# ============================================================================
class PureSFTPipeline:
    """
    Pipeline 2: Fine-tuned 1.5B model without external knowledge.

    Workflow:
      1. User asks a medical question
      2. Fine-tuned model generates answer directly from its weights

    This represents the "scale via fine-tuning" approach — teaching a small model
    medical knowledge through supervised fine-tuning.
    """

    def __init__(self, base_model_id: str = BASE_MODEL_1_5B, lora_path: str = LORA_MODEL_PATH):
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        logger.info(f"Loading Pipeline 2 (Pure SFT) — Base: {self.base_model_id}")

        bnb_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, trust_remote_code=True
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        if os.path.exists(self.lora_path):
            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
            logger.info("LoRA adapters applied.")
        else:
            logger.warning(f"LoRA adapters not found at {self.lora_path}")

        self._model.eval()
        self._loaded = True
        logger.info("Pipeline 2 (Pure SFT) loaded.")

    def answer(self, question: str, max_new_tokens: int = INFERENCE_MAX_NEW_TOKENS) -> dict:
        """
        Answer a medical question using only the fine-tuned model.

        Returns dict with keys: question, answer, pipeline
        """
        self._ensure_loaded()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical AI assistant specializing in cardiology, "
                    "ECG interpretation, and clinical decision support."
                ),
            },
            {"role": "user", "content": question},
        ]

        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=INFERENCE_TEMPERATURE,
                top_p=INFERENCE_TOP_P,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        return {
            "question": question,
            "answer": response.strip(),
            "pipeline": "pure_sft",
        }


# ============================================================================
# Pipeline 3: Hybrid Agent (Fine-Tuned 1.5B + RAG)
# ============================================================================
class HybridAgentPipeline:
    """
    Pipeline 3: Agent-driven hybrid approach combining fine-tuned model + RAG.

    Workflow:
      1. User asks a medical question
      2. Agent decides whether to:
         a. Retrieve knowledge first, then generate (RAG → Model)
         b. Generate directly (Model only)
         c. Retrieve only (RAG only)
      3. Agent executes the chosen action and returns answer

    This represents the "best of both worlds" approach — small efficient model
    augmented with retrieval when needed, orchestrated by an agent.

    Supports two agent implementations:
      - smolagents CodeAgent (if smolagents is installed)
      - Simple rule-based agent (fallback)
    """

    def __init__(self, base_model_id: str = BASE_MODEL_1_5B, lora_path: str = LORA_MODEL_PATH):
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self._rag_tool = None
        self._model_tool = None
        self._agent = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        logger.info("Loading Pipeline 3 (Hybrid Agent)...")

        # Load tools
        from memberC_files.agent.tools import MedicalRAGTool, MedicalModelTool

        logger.info("  Loading RAG tool...")
        self._rag_tool = MedicalRAGTool()

        logger.info("  Loading Model tool...")
        self._model_tool = MedicalModelTool(
            base_model_id=self.base_model_id,
            lora_path=self.lora_path,
        )

        # Try to create smolagents CodeAgent
        try:
            from smolagents import CodeAgent, tool

            @tool
            def execute_medical_knowledge_retrieval(query: str) -> str:
                """Search medical knowledge base for evidence-based information about cardiology,
                ECG interpretation, clinical guidelines, and related topics.
                Args: query — the medical question or topic to search for.
                Returns retrieved medical documents with relevance scores."""
                return self._rag_tool.forward(query)

            @tool
            def invoke_diagnostic_reasoning(prompt: str) -> str:
                """Invoke the fine-tuned medical LLM for diagnostic reasoning, clinical analysis,
                and structured medical response generation.
                Args: prompt — the medical question or clinical scenario to analyze.
                Returns the model's medical analysis and response."""
                return self._model_tool.forward(prompt)

            self._agent = CodeAgent(
                tools=[execute_medical_knowledge_retrieval, invoke_diagnostic_reasoning],
                model=self._model_tool,  # Use medical model as the agent's LLM
                additional_authorized_imports=["json", "re"],
            )
            self._use_smolagents = True
            logger.info("  smolagents CodeAgent created successfully.")

        except ImportError:
            logger.warning("  smolagents not installed. Using rule-based fallback agent.")
            self._use_smolagents = False

        self._loaded = True
        logger.info("Pipeline 3 (Hybrid Agent) loaded.")

    def _rule_based_answer(self, question: str) -> dict:
        """
        Simple rule-based agent logic as fallback when smolagents is unavailable.

        The agent follows this logic:
        1. Always retrieve relevant documents first
        2. Format documents as context
        3. Generate answer with context-augmented prompt
        """
        # Step 1: Retrieve knowledge
        retrieved = self._rag_tool.search(question)

        # Step 2: Build context-augmented prompt
        context_parts = []
        for i, r in enumerate(retrieved, 1):
            context_parts.append(f"[Reference {i}] {r['content']}")
        context = "\n\n".join(context_parts)

        augmented_prompt = f"""You are a medical AI assistant. Use the following reference information to answer the question.

Reference Information:
{context}

Question: {question}

Provide a comprehensive, evidence-based answer:"""

        # Step 3: Generate with fine-tuned model
        answer = self._model_tool.generate(augmented_prompt)

        return {
            "question": question,
            "retrieved_docs": [r["content"] for r in retrieved],
            "retrieval_scores": [r["score"] for r in retrieved],
            "answer": answer,
            "pipeline": "hybrid_agent",
            "agent_type": "rule_based",
        }

    def _smolagents_answer(self, question: str) -> dict:
        """Use smolagents CodeAgent to answer the question."""
        agent_prompt = f"""Answer the following medical question. You can:
1. Use execute_medical_knowledge_retrieval to search for relevant medical information
2. Use invoke_diagnostic_reasoning for clinical analysis

Question: {question}

Provide a thorough, evidence-based answer."""
        result = self._agent.run(agent_prompt)

        return {
            "question": question,
            "answer": str(result),
            "pipeline": "hybrid_agent",
            "agent_type": "smolagents",
        }

    def answer(self, question: str) -> dict:
        """Answer using the hybrid agent pipeline."""
        self._ensure_loaded()

        if self._use_smolagents:
            return self._smolagents_answer(question)
        else:
            return self._rule_based_answer(question)


# ============================================================================
# Unified Agent Runner
# ============================================================================
class MedicalAgentRunner:
    """
    Unified runner that dispatches to the appropriate pipeline based on mode.

    Usage:
        runner = MedicalAgentRunner()
        result = runner.run("What is the treatment for acute MI?", mode=PipelineMode.HYBRID_AGENT)
    """

    def __init__(self):
        self._pipelines = {}

    def _get_pipeline(self, mode: PipelineMode):
        if mode not in self._pipelines:
            if mode == PipelineMode.PURE_RAG:
                self._pipelines[mode] = PureRAGPipeline()
            elif mode == PipelineMode.PURE_SFT:
                self._pipelines[mode] = PureSFTPipeline()
            elif mode == PipelineMode.HYBRID_AGENT:
                self._pipelines[mode] = HybridAgentPipeline()
        return self._pipelines[mode]

    def run(self, question: str, mode: PipelineMode = PipelineMode.HYBRID_AGENT) -> dict:
        """
        Run a medical question through the specified pipeline.

        Args:
            question: Medical question or clinical scenario
            mode: Which pipeline to use

        Returns:
            Dict with answer and pipeline metadata
        """
        pipeline = self._get_pipeline(mode)
        return pipeline.answer(question)

    def run_all_pipelines(self, question: str) -> dict:
        """
        Run the same question through ALL three pipelines for comparison.

        Returns:
            Dict with key=pipeline_name, value=result_dict
        """
        results = {}
        for mode in PipelineMode:
            try:
                results[mode.value] = self.run(question, mode)
            except Exception as e:
                logger.error(f"Pipeline {mode.value} failed: {e}")
                results[mode.value] = {"error": str(e), "pipeline": mode.value}
        return results


# ============================================================================
# CLI Entry Point
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Medical AI Agent")
    parser.add_argument("--question", "-q", type=str, required=True,
                        help="Medical question to answer")
    parser.add_argument("--mode", "-m", type=str, default="hybrid",
                        choices=["pure_rag", "pure_sft", "hybrid", "all"],
                        help="Pipeline mode (default: hybrid)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    runner = MedicalAgentRunner()

    if args.mode == "all":
        results = runner.run_all_pipelines(args.question)
    else:
        mode = PipelineMode(args.mode)
        results = runner.run(args.question, mode)

    # Output
    output_str = json.dumps(results, ensure_ascii=False, indent=2)
    print(output_str)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
