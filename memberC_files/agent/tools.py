"""
Medical Agent Tools — RAG Knowledge Retrieval & Model Inference.

This module provides two core tools that the Medical AI Agent uses:
  1. MedicalRAGTool  — FAISS-based semantic search over 78K cleaned medical QA pairs
  2. MedicalModelTool — QLoRA fine-tuned DeepSeek-R1-Distill-Qwen-1.5B inference

Both tools follow the smolagents Tool interface and expose a `forward` method.
"""
import os
import sys
import pickle
import logging
import numpy as np
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memberC_files.config.settings import (
    FAISS_INDEX_PATH,
    FAISS_DOCS_PATH,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    BASE_MODEL_1_5B,
    LORA_MODEL_PATH,
    QUANTIZATION_CONFIG,
    INFERENCE_MAX_NEW_TOKENS,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P,
    INFERENCE_REPETITION_PENALTY,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Tool 1: Medical RAG Knowledge Retrieval
# ============================================================================
class MedicalRAGTool:
    """
    FAISS-based semantic search over a corpus of 78,136 cleaned medical QA pairs.

    Uses BAAI/bge-m3 embeddings (optimal config from Member A's ablation study:
    chunk_size=512, chunk_overlap=50, top_k=3, achieving F1=0.8628).

    Usage (standalone):
        >>> rag = MedicalRAGTool()
        >>> docs = rag.search("房缺 ASD 封堵术后长期预后如何？")
    """

    def __init__(
        self,
        index_path: str = FAISS_INDEX_PATH,
        docs_path: str = FAISS_DOCS_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = RAG_TOP_K,
    ):
        self.index_path = index_path
        self.docs_path = docs_path
        self.top_k = top_k
        self.embedding_model_name = embedding_model
        self._index = None
        self._docs = None
        self._embedder = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "medical_knowledge_retrieval"

    @property
    def description(self) -> str:
        return (
            "Search a medical knowledge base of 78,136 cleaned medical QA pairs "
            "using semantic similarity. Use this tool to retrieve evidence-based "
            "medical information about cardiology, ECG interpretation, clinical "
            "decision support, and related topics. Returns the top relevant documents."
        )

    def _ensure_loaded(self):
        """Lazy-load the FAISS index, document store, and embedding model."""
        if self._loaded:
            return
        logger.info("Loading Medical RAG components...")

        # Load FAISS index if it exists; otherwise, raise with instructions
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run 'python -m memberC_files.agent.build_index' to generate it."
            )

        import faiss
        logger.info(f"  Loading FAISS index from {self.index_path}...")
        self._index = faiss.read_index(self.index_path)

        logger.info(f"  Loading document store from {self.docs_path}...")
        with open(self.docs_path, "rb") as f:
            self._docs = pickle.load(f)
        logger.info(f"  {len(self._docs)} documents loaded.")

        logger.info(f"  Loading embedding model: {self.embedding_model_name}...")
        self._embedder = SentenceTransformer(self.embedding_model_name)
        logger.info("  Embedding model loaded.")

        self._loaded = True

    def search(self, query: str, top_k: Optional[int] = None) -> list:
        """
        Search the medical knowledge base for documents relevant to the query.

        Args:
            query: Natural language medical question
            top_k: Number of documents to retrieve (defaults to RAG_TOP_K=3)

        Returns:
            List of retrieved document content strings
        """
        import faiss  # local import to avoid error on import if faiss missing

        if top_k is None:
            top_k = self.top_k
        self._ensure_loaded()

        # Encode query
        query_vector = self._embedder.encode([query], show_progress_bar=False)
        query_vector = np.array(query_vector).astype("float32")

        # Normalize for cosine similarity (if using inner-product index)
        faiss.normalize_L2(query_vector)

        # Search
        distances, indices = self._index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self._docs):
                doc = self._docs[idx]
                results.append({
                    "content": doc.get("content", ""),
                    "score": float(dist),
                    "source": doc.get("metadata", {}).get("dataset", "Unknown"),
                })

        logger.info(
            f"RAG search: '{query[:80]}...' → {len(results)} results (top score: {results[0]['score']:.4f})"
        )
        return results

    def forward(self, query: str, **kwargs) -> str:
        """
        smolagents-compatible forward method.
        Accepts and ignores extra kwargs from smolagents.

        Returns retrieved documents formatted as a single context string.
        """
        results = self.search(query)
        if not results:
            return "[No relevant medical documents found.]"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"--- Document {i} (relevance: {r['score']:.4f}, source: {r['source']}) ---")
            lines.append(r["content"])
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# Tool 2: Fine-Tuned Medical Model Inference
# ============================================================================
class MedicalModelTool:
    """
    QLoRA fine-tuned DeepSeek-R1-Distill-Qwen-1.5B medical model inference.

    Loads the base model with 4-bit NF4 quantization and applies LoRA adapters
    trained by Member B on 10,000 cardiology/ECG QA pairs (including 2,000 CoT samples).

    Usage (standalone):
        >>> model = MedicalModelTool()
        >>> answer = model.generate("患者 ST 段抬高 3mm，胸痛，请诊断。")
    """

    def __init__(
        self,
        base_model_id: str = BASE_MODEL_1_5B,
        lora_path: str = LORA_MODEL_PATH,
        use_4bit: bool = True,
        max_new_tokens: int = INFERENCE_MAX_NEW_TOKENS,
        temperature: float = INFERENCE_TEMPERATURE,
        top_p: float = INFERENCE_TOP_P,
    ):
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Set up quantization config
        self.bnb_config = None
        if use_4bit:
            self.bnb_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)

    @property
    def name(self) -> str:
        return "diagnostic_reasoning_model"

    @property
    def description(self) -> str:
        return (
            "Invoke a fine-tuned medical LLM (DeepSeek-R1-Distill-Qwen-1.5B with QLoRA) "
            "specialized in cardiology, ECG interpretation, and clinical decision support. "
            "Use this tool for diagnostic reasoning, clinical analysis, and generating "
            "structured medical responses. The model was fine-tuned on 10,000 medical QA pairs."
        )

    def _ensure_loaded(self):
        """Lazy-load the model, tokenizer, and LoRA adapters."""
        if self._loaded:
            return
        logger.info("Loading Medical Model (this may take 2-5 minutes on first load)...")

        # Load tokenizer
        logger.info(f"  Loading tokenizer: {self.base_model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "right"

        # Load base model with quantization
        logger.info(f"  Loading base model: {self.base_model_id}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Apply LoRA adapters
        if os.path.exists(self.lora_path):
            from peft import PeftModel
            logger.info(f"  Applying LoRA adapters from: {self.lora_path}")
            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
            logger.info("  LoRA adapters applied successfully.")
        else:
            logger.warning(
                f"  LoRA adapters not found at {self.lora_path}. "
                "Using base model without fine-tuning."
            )

        self._model.eval()
        self._loaded = True
        logger.info("Medical Model loaded and ready.")

    def _build_messages(self, prompt: str) -> list:
        """Build system + user messages for the chat template."""
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert medical AI assistant specializing in cardiology, "
                    "ECG interpretation, and clinical decision support. Provide accurate, "
                    "evidence-based, and structured medical information."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, **kwargs) -> str:
        """
        Generate a medical response for the given prompt.

        Args:
            prompt: Medical question or clinical scenario
            max_new_tokens: Override default max tokens

        Returns:
            Generated medical response string
        """
        self._ensure_loaded()
        max_tokens = max_new_tokens or self.max_new_tokens

        messages = self._build_messages(prompt)
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=INFERENCE_REPETITION_PENALTY,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def forward(self, prompt: str, **kwargs) -> str:
        """
        smolagents-compatible forward method. Accepts and ignores extra kwargs
        (e.g. stop_sequences) from smolagents CodeAgent.
        """
        return self.generate(prompt)


# ============================================================================
# Tool Factory — Build smolagents-compatible Tool instances
# ============================================================================
def create_rag_tool() -> MedicalRAGTool:
    """Factory for creating a MedicalRAGTool instance (singleton-style in agent)."""
    return MedicalRAGTool()


def create_model_tool() -> MedicalModelTool:
    """Factory for creating a MedicalModelTool instance (singleton-style in agent)."""
    return MedicalModelTool()


# ============================================================================
# Standalone test
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Medical Agent Tools...")
    print("=" * 60)

    # Test RAG Tool
    print("\n[1] Testing MedicalRAGTool...")
    try:
        rag = MedicalRAGTool()
        results = rag.search(
            "What are the long-term outcomes of atrial septal defect closure in adults?"
        )
        for r in results:
            print(f"  Score: {r['score']:.4f} | Source: {r['source']}")
            print(f"  Content: {r['content'][:200]}...")
            print()
    except Exception as e:
        print(f"  RAG tool test skipped: {e}")

    # Test Model Tool
    print("\n[2] Testing MedicalModelTool...")
    try:
        model = MedicalModelTool()
        response = model.generate(
            "Patient presents with 3mm ST elevation in V1-V4. Suspected diagnosis?"
        )
        print(f"  Response: {response[:300]}...")
    except Exception as e:
        print(f"  Model tool test skipped: {e}")

    print("\nDone.")
