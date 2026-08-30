"""
Test Dataset Extraction for RAGAS Evaluation.

Extracts 100-200 held-out medical QA samples that were NOT used in Member B's
SFT training dataset. These serve as the evaluation benchmark for comparing
the three pipelines.

Strategy:
  1. Load the full RAG corpus (78,136 items)
  2. Load the SFT training dataset (10,000 items) to identify training samples
  3. Filter out training samples by content hash
  4. Stratified sample across 5 medical domains
  5. Save evaluation dataset with ground truth answers
"""
import os
import sys
import json
import hashlib
import random
import logging
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from memberC_files.config.settings import (
    RAG_CORPUS_PATH,
    SFT_DATASET_PATH,
    DATA_DIR,
    EVAL_TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Medical domains for stratified sampling
# ============================================================================
DOMAIN_CATEGORIES = {
    "cardiology_basic": [
        "Cardiology Knowledge (Basic Knowledge QA)",
    ],
    "cross_modal_ecg": [
        "Cross modal diagnosis (ECG Text corresponding diagnosis)",
        "ECG",
        "electrocardiogram",
    ],
    "complex_diagnosis": [
        "Complex diagnosis (analysis of difficult diseases)",
        "Long text diagnosis",
    ],
    "risk_assessment": [
        "Generate risk assessment",
    ],
    "clinical_dialogue": [
        "Medical inquiry, multiple rounds of dialogue between doctors and patients",
    ],
}


def content_hash(content: str) -> str:
    """Generate a hash of the content for deduplication."""
    # Normalize: lowercase, strip extra whitespace
    normalized = " ".join(content.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def load_training_hashes(sft_path: str) -> set:
    """Extract question hashes from the SFT training dataset."""
    logger.info(f"Loading SFT training dataset: {sft_path}")
    with open(sft_path, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    hashes = set()
    for item in sft_data:
        # Extract user question from conversations
        for conv in item.get("conversations", []):
            if conv["role"] == "user":
                hashes.add(content_hash(conv["content"]))
    logger.info(f"  Extracted {len(hashes)} unique question hashes from training set.")
    return hashes


def extract_test_samples(
    corpus_path: str,
    training_hashes: set,
    target_size: int = EVAL_TEST_SIZE,
    seed: int = 42,
) -> list:
    """
    Extract held-out test samples from the corpus.

    Filters out any sample whose question matches a training example.
    Uses stratified sampling across domains for representativeness.
    """
    random.seed(seed)

    logger.info(f"Loading RAG corpus: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info(f"  Total corpus size: {len(corpus)}")

    # Parse and bucket by domain
    domain_buckets = defaultdict(list)

    for item in corpus:
        content = item.get("content", "")
        metadata = item.get("metadata", {})
        dataset_name = metadata.get("dataset", "Unknown")

        # Parse Question and Answer
        q_match = None
        for sep in ["\nAnswer:", "\nAnswer:", "Question:\n"]:
            # Try simple split
            if "Question:" in content and "Answer:" in content:
                # Extract question part
                q_start = content.find("Question:") + len("Question:")
                a_start = content.find("Answer:")
                question = content[q_start:a_start].strip() if a_start > q_start else content
                answer = content[a_start + len("Answer:"):].strip()
                q_match = True
                break

        if not q_match:
            question = content[:200]  # Fallback: first 200 chars as question
            answer = content[200:]

        # Skip if this question was in training set
        q_hash = content_hash(question)
        if q_hash in training_hashes:
            continue

        # Classify domain
        domain = "general"
        for cat, keywords in DOMAIN_CATEGORIES.items():
            for kw in keywords:
                if kw.lower() in dataset_name.lower():
                    domain = cat
                    break

        domain_buckets[domain].append({
            "question": question,
            "answer": answer.strip(),
            "dataset": dataset_name,
            "domain": domain,
            "source_id": item.get("id", "unknown"),
            "full_content": content,
        })

    logger.info("Domain distribution (available):")
    for domain, samples in sorted(domain_buckets.items()):
        logger.info(f"  {domain}: {len(samples)}")

    # Stratified sampling
    samples_per_domain = target_size // len(domain_buckets)
    remainder = target_size - samples_per_domain * len(domain_buckets)

    selected = []
    domain_order = sorted(domain_buckets.keys())

    for i, domain in enumerate(domain_order):
        bucket = domain_buckets[domain]
        n = samples_per_domain + (1 if i < remainder else 0)
        n = min(n, len(bucket))
        sampled = random.sample(bucket, n) if n > 0 else []
        selected.extend(sampled)
        logger.info(f"  Selected {n} from '{domain}'")

    # If we didn't reach target, fill from remaining
    if len(selected) < target_size:
        all_remaining = []
        for domain in domain_order:
            already_ids = {s["source_id"] for s in selected}
            remaining = [s for s in domain_buckets[domain] if s["source_id"] not in already_ids]
            all_remaining.extend(remaining)
        random.shuffle(all_remaining)
        needed = target_size - len(selected)
        extra = all_remaining[:needed]
        selected.extend(extra)
        logger.info(f"  Filled {len(extra)} extra samples to reach target.")

    random.shuffle(selected)
    logger.info(f"Final test dataset: {len(selected)} samples")
    return selected


def save_test_dataset(samples: list, output_path: str):
    """Save the extracted test dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Test dataset saved to: {output_path}")

    # Print domain distribution
    domain_counts = defaultdict(int)
    for s in samples:
        domain_counts[s["domain"]] += 1
    logger.info("Final domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain}: {count} ({100*count/len(samples):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Extract RAGAS evaluation test dataset")
    parser.add_argument("--target-size", type=int, default=EVAL_TEST_SIZE,
                        help=f"Target number of test samples (default: {EVAL_TEST_SIZE})")
    parser.add_argument("--corpus", type=str, default=RAG_CORPUS_PATH)
    parser.add_argument("--sft-dataset", type=str, default=SFT_DATASET_PATH)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(DATA_DIR, "eval_test_dataset.json")

    # Step 1: Load training hashes for dedup
    train_hashes = load_training_hashes(args.sft_dataset)

    # Step 2: Extract held-out samples
    samples = extract_test_samples(args.corpus, train_hashes, args.target_size, args.seed)

    # Step 3: Save
    save_test_dataset(samples, args.output)

    # Step 4: Print sample preview
    print("\n" + "=" * 60)
    print("Sample Preview:")
    print("=" * 60)
    for i, s in enumerate(samples[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Domain: {s['domain']} | Source: {s['dataset']}")
        print(f"Q: {s['question'][:150]}...")
        print(f"A: {s['answer'][:200]}...")


if __name__ == "__main__":
    main()
