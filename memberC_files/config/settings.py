"""
Project-wide configuration constants for the Medical AI Agent system.
Centralizes all paths, model names, and hyperparameters.
"""
import os

# ============================================================================
# Paths (relative to project root)
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Member A assets
MEMBER_A_DIR = os.path.join(PROJECT_ROOT, "memberA_files")
RAG_CORPUS_PATH = os.path.join(MEMBER_A_DIR, "data/cleaned/rag_corpus.json")
FAISS_INDEX_PATH = os.path.join(MEMBER_A_DIR, "models/medical_rag_index.faiss")
FAISS_DOCS_PATH = os.path.join(MEMBER_A_DIR, "models/medical_rag_index_docs.pkl")
FAISS_EMBEDDINGS_PATH = os.path.join(MEMBER_A_DIR, "models/medical_rag_index_embeddings.npy")

# Member B assets
MEMBER_B_DIR = os.path.join(PROJECT_ROOT, "memberB_files")
LORA_MODEL_PATH = os.path.join(MEMBER_B_DIR, "models/deepseek-1.5b-medical-lora")
SFT_DATASET_PATH = os.path.join(MEMBER_B_DIR, "data/sft_train_dataset.json")

# Member C outputs
MEMBER_C_DIR = os.path.join(PROJECT_ROOT, "memberC_files")
OUTPUT_DIR = os.path.join(MEMBER_C_DIR, "outputs")
DATA_DIR = os.path.join(MEMBER_C_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================================
# Model & Embedding Config
# ============================================================================
# Base model (used for Pipeline 1: Pure RAG with 7B)
BASE_MODEL_7B = "Qwen/Qwen2-7B-Instruct"

# Fine-tuned model (used for Pipeline 2: Pure SFT and Pipeline 3: Hybrid Agent)
BASE_MODEL_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Embedding model for RAG
EMBEDDING_MODEL = "BAAI/bge-m3"

# ============================================================================
# RAG Configuration (from Member A's ablation study - optimal params)
# ============================================================================
RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 3

# ============================================================================
# Inference Configuration
# ============================================================================
INFERENCE_MAX_NEW_TOKENS = 512
INFERENCE_TEMPERATURE = 0.6
INFERENCE_TOP_P = 0.9
INFERENCE_REPETITION_PENALTY = 1.1

# ============================================================================
# Quantization Configuration (for QLoRA-based models)
# ============================================================================
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
}

# ============================================================================
# Evaluation Configuration
# ============================================================================
EVAL_TEST_SIZE = 150  # Number of test samples for RAGAS evaluation
EVAL_METRICS = ["faithfulness", "answer_relevancy"]

# ============================================================================
# Agent Configuration
# ============================================================================
AGENT_SYSTEM_PROMPT = """You are an expert medical AI assistant specializing in cardiology,
ECG interpretation, and clinical decision support. You have access to:
1. A medical knowledge retrieval tool for evidence-based information
2. A diagnostic reasoning model fine-tuned on medical data

Always provide accurate, evidence-based, and structured medical information.
When using tools, explain your reasoning step by step."""
