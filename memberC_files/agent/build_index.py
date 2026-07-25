"""
Build FAISS vector index from the cleaned RAG corpus.

This script processes the 78,136 medical QA pairs from Member A's pipeline,
chunks them, generates BAAI/bge-m3 embeddings, and saves a FAISS index for
semantic search. Uses the optimal configuration from Member A's ablation study:
chunk_size=512, chunk_overlap=50, top_k=3 (F1=0.8628).

Usage:
    python -m memberC_files.agent.build_index
    python -m memberC_files.agent.build_index --chunk-size 512 --chunk-overlap 50
"""
import os
import sys
import json
import pickle
import argparse
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from memberC_files.config.settings import (
    RAG_CORPUS_PATH,
    FAISS_INDEX_PATH,
    FAISS_DOCS_PATH,
    FAISS_EMBEDDINGS_PATH,
    EMBEDDING_MODEL,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str) -> list:
    """Load the raw RAG corpus."""
    logger.info(f"Loading corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} items from corpus.")
    return data


def chunk_document(content: str, chunk_size: int, overlap: int) -> list:
    """
    Split a document into overlapping chunks by approximate word count.

    The chunk_size and overlap parameters control the granularity of retrieval.
    Smaller chunks give more precise results; larger chunks provide more context.
    """
    words = content.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


def build_document_chunks(corpus: list, chunk_size: int, overlap: int) -> tuple:
    """
    Chunk all documents and return (chunk_texts, chunk_metadata).
    """
    texts = []
    metadata = []

    for item in corpus:
        content = item.get("content", "")
        chunks = chunk_document(content, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "id": f"{item.get('id', 'unknown')}_chunk{i}",
                "content": chunk,
                "metadata": item.get("metadata", {}),
            })

    logger.info(f"Chunked {len(corpus)} documents into {len(texts)} chunks "
                f"(chunk_size={chunk_size}, overlap={overlap})")
    return texts, metadata


def build_embeddings(texts: list, model_name: str, batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for all text chunks using SentenceTransformers.

    Uses BAAI/bge-m3 which produces 1024-dimensional normalized embeddings.
    """
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )
    logger.info(f"Generated embeddings matrix: {embeddings.shape}")
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray, index_path: str):
    """
    Build a FAISS IVF-Flat index for efficient similarity search.

    IVF-Flat balances speed and accuracy: clusters embeddings and searches
    only the nearest clusters for fast approximate nearest neighbor lookup.

    For 148K+ vectors, IVF with ~nlist=sqrt(n) provides good trade-off.
    """
    import faiss

    dim = embeddings.shape[1]
    n_vectors = embeddings.shape[0]

    # nlist = sqrt(n_vectors) is a good heuristic, capped at 4096
    nlist = min(int(np.sqrt(n_vectors)), 4096)
    # Ensure nlist is reasonable
    nlist = max(nlist, 100)

    logger.info(f"Building FAISS IVF-Flat index: dim={dim}, n_vectors={n_vectors}, nlist={nlist}")

    # Quantizer for coarse clustering
    quantizer = faiss.IndexFlatIP(dim)  # Inner Product (cosine similarity on normalized vectors)

    # IVF-Flat index
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train the index (required for IVF)
    logger.info("Training IVF index...")
    index.train(embeddings)

    # Add vectors
    logger.info("Adding vectors to index...")
    index.add(embeddings)

    # Save
    logger.info(f"Saving FAISS index to: {index_path}")
    faiss.write_index(index, index_path)

    logger.info(f"Index built: {index.ntotal} vectors, dimension {dim}")
    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for Medical RAG")
    parser.add_argument("--chunk-size", type=int, default=RAG_CHUNK_SIZE,
                        help=f"Chunk size in words (default: {RAG_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=RAG_CHUNK_OVERLAP,
                        help=f"Chunk overlap in words (default: {RAG_CHUNK_OVERLAP})")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedding batch size (default: 32)")
    parser.add_argument("--corpus", type=str, default=RAG_CORPUS_PATH)
    args = parser.parse_args()

    # Step 1: Load corpus
    corpus = load_corpus(args.corpus)

    # Step 2: Chunk documents
    texts, metadata = build_document_chunks(corpus, args.chunk_size, args.chunk_overlap)

    # Step 3: Generate embeddings
    embeddings = build_embeddings(texts, EMBEDDING_MODEL, args.batch_size)

    # Step 4: Save embeddings
    logger.info(f"Saving embeddings to: {FAISS_EMBEDDINGS_PATH}")
    os.makedirs(os.path.dirname(FAISS_EMBEDDINGS_PATH), exist_ok=True)
    np.save(FAISS_EMBEDDINGS_PATH, embeddings)

    # Step 5: Build and save FAISS index
    build_faiss_index(embeddings, FAISS_INDEX_PATH)

    # Step 6: Save document metadata
    logger.info(f"Saving document metadata to: {FAISS_DOCS_PATH}")
    with open(FAISS_DOCS_PATH, "wb") as f:
        pickle.dump(metadata, f)

    # Step 7: Summary
    logger.info("=" * 60)
    logger.info("FAISS Index Build Complete!")
    logger.info(f"  Documents: {len(corpus)}")
    logger.info(f"  Chunks: {len(texts)}")
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    logger.info(f"  Index size: {os.path.getsize(FAISS_INDEX_PATH) / 1024**2:.1f} MB")
    logger.info(f"  Docs store: {os.path.getsize(FAISS_DOCS_PATH) / 1024**2:.1f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
