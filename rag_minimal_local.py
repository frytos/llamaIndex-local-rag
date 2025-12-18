#!/usr/bin/env python3
# Minimal local-only RAG: PDF + DOCX -> chunk -> embed -> save -> retrieve -> (optional) local LLM answer
# Storage format:
#   store_dir/
#     chunks.jsonl
#     embeddings.npy
#     config.json

from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import numpy as np

# -------- Parsers --------

def read_pdf(path: Path) -> str:
    # PyMuPDF (fitz)
    import fitz  # pip install pymupdf
    doc = fitz.open(str(path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)

def read_docx(path: Path) -> str:
    # python-docx
    from docx import Document  # pip install python-docx
    d = Document(str(path))
    # paragraphs are the primary block-level text objects in Word docs
    return "\n".join(p.text for p in d.paragraphs)

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def load_document_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext in (".txt", ".md"):
        return read_txt(path)
    raise ValueError(f"Unsupported file type: {ext} ({path})")

# -------- Chunking --------

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    step = max(1, chunk_chars - overlap)
    while i < n:
        chunk = text[i : i + chunk_chars]
        chunks.append(chunk)
        i += step
    return chunks

# -------- Embeddings --------

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine sim = dot product
    )
    return np.asarray(emb, dtype=np.float32)

# -------- Store format --------

@dataclass
class StoreConfig:
    embedding_model: str
    chunk_chars: int
    overlap: int

def store_paths(store_dir: Path) -> Dict[str, Path]:
    return {
        "config": store_dir / "config.json",
        "chunks": store_dir / "chunks.jsonl",
        "embeddings": store_dir / "embeddings.npy",
    }

def write_store(store_dir: Path, cfg: StoreConfig, chunks_meta: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    p = store_paths(store_dir)

    p["config"].write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    with p["chunks"].open("w", encoding="utf-8") as f:
        for row in chunks_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    np.save(p["embeddings"], embeddings)

def read_store(store_dir: Path) -> Tuple[StoreConfig, List[Dict[str, Any]], np.ndarray]:
    p = store_paths(store_dir)
    if not p["config"].exists() or not p["chunks"].exists() or not p["embeddings"].exists():
        raise FileNotFoundError(f"Store incomplete in: {store_dir}")

    cfg = StoreConfig(**json.loads(p["config"].read_text(encoding="utf-8")))
    chunks_meta = []
    with p["chunks"].open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks_meta.append(json.loads(line))

    embeddings = np.load(p["embeddings"])
    return cfg, chunks_meta, embeddings

# -------- Retrieval --------

def top_k_cosine(query_emb: np.ndarray, doc_embs: np.ndarray, k: int = 4) -> List[int]:
    # embeddings are normalized -> cosine similarity = dot product
    sims = doc_embs @ query_emb
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, kth=k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()

# -------- Local LLM (optional) --------

def try_llama_answer(model_path: str, prompt: str, max_tokens: int = 400, temperature: float = 0.2) -> str | None:
    try:
        from llama_cpp import Llama
    except Exception:
        return None

    # Reasonable defaults for M1; tweak if you want
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=max(2, os.cpu_count() or 4),
        n_gpu_layers=999,  # let Metal take what it can
        verbose=False,
    )

    # Try chat API if available, otherwise fallback to completion
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If missing, say you don't know."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>"],
        )
        return out["choices"][0]["text"].strip()

# -------- Commands --------

def cmd_index(args: argparse.Namespace) -> None:
    docs = [Path(p).expanduser().resolve() for p in args.docs]
    for d in docs:
        if not d.exists():
            raise FileNotFoundError(d)

    all_chunks: List[str] = []
    chunks_meta: List[Dict[str, Any]] = []

    for doc_path in docs:
        text = load_document_text(doc_path)
        chunks = chunk_text(text, chunk_chars=args.chunk_chars, overlap=args.overlap)
        for j, ch in enumerate(chunks):
            chunks_meta.append(
                {
                    "id": len(chunks_meta),
                    "source_path": str(doc_path),
                    "chunk_index": j,
                    "text": ch,
                }
            )
            all_chunks.append(ch)

    if not all_chunks:
        raise RuntimeError("No text extracted; check your documents.")

    embeddings = embed_texts(all_chunks, model_name=args.embedding_model)

    cfg = StoreConfig(
        embedding_model=args.embedding_model,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )
    write_store(Path(args.store_dir), cfg, chunks_meta, embeddings)
    print(f"✅ Indexed {len(docs)} docs into {len(all_chunks)} chunks at: {args.store_dir}")

def cmd_ask(args: argparse.Namespace) -> None:
    store_dir = Path(args.store_dir).expanduser().resolve()
    cfg, chunks_meta, doc_embs = read_store(store_dir)

    # Embed query with same model
    q_emb = embed_texts([args.question], model_name=cfg.embedding_model)[0]

    idxs = top_k_cosine(q_emb, doc_embs, k=args.k)
    retrieved = [chunks_meta[i] for i in idxs]

    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[source: {r['source_path']} | chunk: {r['chunk_index']}]\n{r['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"QUESTION:\n{args.question}\n\n"
        f"CONTEXT (use only this):\n{context}\n\n"
        f"Answer in a clear, direct way and cite sources like [source: ... | chunk: ...]."
    )

    if args.llm_model_path:
        ans = try_llama_answer(args.llm_model_path, prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        if ans is None:
            print("⚠️ llama-cpp-python not available. Printing retrieved context only.\n")
            print(context)
        else:
            print(ans)
    else:
        # No LLM: show context + “manual” answer hint
        print("No LLM configured (--llm-model-path). Here are the top retrieved chunks:\n")
        print(context)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal local-only RAG (PDF + DOCX) with on-disk index.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Parse docs, chunk, embed, and write store.")
    p_index.add_argument("--docs", nargs="+", required=True, help="Paths to documents (.pdf, .docx, .txt, .md).")
    p_index.add_argument("--store-dir", required=True, help="Directory to write the index.")
    p_index.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformers model name.")
    p_index.add_argument("--chunk-chars", type=int, default=1200)
    p_index.add_argument("--overlap", type=int, default=200)
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Query an existing store.")
    p_ask.add_argument("--store-dir", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--k", type=int, default=4)
    p_ask.add_argument("--llm-model-path", default="", help="Path to a local GGUF model (optional).")
    p_ask.add_argument("--max-tokens", type=int, default=400)
    p_ask.add_argument("--temperature", type=float, default=0.2)
    p_ask.set_defaults(func=cmd_ask)

    return p

def main() -> None:
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
