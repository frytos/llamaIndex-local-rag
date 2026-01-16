# RAG Evaluation Framework

**Created:** 2026-01-16
**Purpose:** Systematic evaluation of RAG pipeline quality and performance

---

## Current RAG Configuration

### Vector Database
- **Technology:** PostgreSQL 14+ with pgvector extension
- **Indexing:** HNSW indices (100x+ faster queries)
- **Similarity Metric:** Cosine similarity
- **Storage:** Persistent vector storage with metadata

### Chunking Strategy
- **Default Chunk Size:** 700 characters
- **Default Overlap:** 150 characters (21.4% overlap ratio)
- **Presets Available:**
  - Chat messages: 300/50
  - Short docs: 500/100
  - General documents: 700/150 (default)
  - Technical docs: 1000/200
  - Long-form: 1500/300
- **Strategy:** Sentence-aware splitting with overlap preservation

### Embedding Models
- **Primary:** BAAI/bge-small-en (384 dimensions)
- **Multilingual:** BAAI/bge-m3 (1024 dimensions)
- **Alternative:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Backend:** HuggingFace (CPU/GPU) or MLX (Apple Silicon)
- **GPU Acceleration:** RunPod RTX 4090 (100x speedup)

### Reranker
- **Status:** ✅ YES - Implemented
- **Model:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Strategy:** Re-ranks retrieved chunks for better relevance
- **Configurable:** Can enable/disable per query

### Metadata Extraction
- **Enhanced Metadata:** ✅ Enabled
- **Extracted:** Dates, emails, URLs, content types, key phrases
- **Filtering:** Metadata-based filtering supported
- **Provenance:** Source tracking for citations

### File Types Supported
- **Documents:** PDF, HTML, MD, TXT
- **Code:** PY, JS, JSON
- **Data:** CSV
- **Conversational:** HTML (Facebook Messenger exports)

### Retrieval Configuration
- **Top-K:** 4-10 chunks (configurable)
- **Hybrid Search:** Vector + BM25 (alpha configurable)
- **MMR Diversity:** Optional diversity filtering
- **Semantic Cache:** Query result caching (92% similarity threshold)

---

## Evaluation Framework

### Test Packs (Based on Your Setup)

#### Pack 1: Technical Documentation
**Best fit for:**
- Code documentation
- API references
- Technical guides

**Query Types:**
- Direct lookups ("What is the embed_nodes function?")
- Multi-step ("How do I configure PostgreSQL SSL?")
- Comparative ("Difference between bge-small and bge-m3?")

---

#### Pack 2: Personal Documents
**Best fit for:**
- PDFs (contracts, invoices, letters)
- Personal notes
- Mixed content types

**Query Types:**
- Entity extraction ("Who is mentioned in document X?")
- Date-based ("What happened in March 2022?")
- Cross-document ("Find all references to topic Y")

---

#### Pack 3: Conversational Data
**Best fit for:**
- Chat logs (Messenger, Slack)
- Email threads
- Timestamped conversations

**Query Types:**
- Person-specific ("What did Alice say about the project?")
- Timeline ("Conversations from last week")
- Topic tracking ("All mentions of vacation plans")

---

## Evaluation Metrics

### 1. Answer Grounded (Citation Quality)
**Score 0-10:**
- 10: Every claim has specific chunk citation
- 7-9: Most claims cited, minor details uncited
- 4-6: Some citations, some invented
- 0-3: Mostly hallucinated, few citations

**Auto-check:**
```python
def check_grounding(answer, retrieved_chunks):
    # Count factual claims in answer
    # Verify each claim appears in retrieved chunks
    # Calculate citation coverage %
```

---

### 2. Source Selection (Relevance)
**Score 0-10:**
- 10: Retrieved most relevant chunks, ignored noise
- 7-9: Good chunks, some irrelevant included
- 4-6: Missed key chunks or included too much noise
- 0-3: Retrieved wrong sections entirely

**Auto-check:**
```python
def check_source_quality(retrieved_chunks, expected_chunks):
    # Precision: % of retrieved that are relevant
    # Recall: % of expected chunks that were retrieved
    # F1 score
```

---

### 3. Completeness
**Score 0-10:**
- 10: Covers all aspects of query
- 7-9: Covers main points, minor gaps
- 4-6: Partial answer, missing key points
- 0-3: Incomplete, major omissions

**Manual check:** Compare answer to ground truth

---

### 4. No Invention (Hallucination Resistance)
**Score 0-10:**
- 10: Never makes up information, says "not found" when appropriate
- 7-9: Rare invention, mostly accurate
- 4-6: Some hallucination mixed with facts
- 0-3: Frequent invention

**Auto-check:**
```python
def check_hallucination(answer, source_text):
    # Extract factual claims
    # Verify each claim in source
    # Flag unverifiable claims
```

---

### 5. Multi-Hop Reasoning
**Score 0-10:**
- 10: Correctly combines info from multiple chunks/docs
- 7-9: Mostly correct multi-hop, minor errors
- 4-6: Struggles with multi-hop, some success
- 0-3: Can't combine info across chunks

**Test queries:**
- "Compare X and Y" (requires 2+ chunks)
- "Who said what about Z?" (requires linking)
- "Timeline of events" (requires ordering)

---

### 6. Latency
**Score 0-10:**
- 10: < 2 seconds end-to-end
- 7-9: 2-5 seconds
- 4-6: 5-10 seconds
- 0-3: > 10 seconds

**Auto-measure:**
```python
import time
start = time.time()
answer = query_engine.query(question)
latency = time.time() - start
```

---

## Test Harness Structure

### JSON Format

```json
{
  "test_pack_name": "Technical Documentation",
  "description": "Tests for code and technical content",
  "test_cases": [
    {
      "id": "tech-001",
      "query": "How do I configure GPU embeddings?",
      "expected_sources": [
        "services/embedding_service.py",
        "utils/runpod_embedding_client.py"
      ],
      "expected_citations": [
        "RUNPOD_EMBEDDING_API_KEY",
        "port 8001",
        "GPU acceleration"
      ],
      "grading_rules": {
        "must_mention": ["FastAPI", "RunPod", "GPU"],
        "should_not_mention": ["CPU-only", "local embedding"],
        "expected_steps": [
          "Set RUNPOD_EMBEDDING_API_KEY",
          "Start embedding service",
          "Verify GPU available"
        ]
      },
      "type": "how-to",
      "difficulty": "medium",
      "multi_hop": false
    },
    {
      "id": "tech-002",
      "query": "What's the difference between bge-small and bge-m3 models?",
      "expected_sources": [
        "CLAUDE.md",
        "rag_low_level_m1_16gb_verbose.py"
      ],
      "expected_citations": [
        "384 dimensions",
        "1024 dimensions",
        "multilingual"
      ],
      "grading_rules": {
        "must_mention": ["dimensions", "384", "1024"],
        "comparison_aspects": ["size", "speed", "language support"],
        "should_cite": true
      },
      "type": "comparison",
      "difficulty": "easy",
      "multi_hop": true
    }
  ],
  "overall_scoring": {
    "grounding_weight": 0.25,
    "source_selection_weight": 0.20,
    "completeness_weight": 0.20,
    "no_invention_weight": 0.15,
    "multi_hop_weight": 0.10,
    "latency_weight": 0.10
  }
}
```

---

## Evaluation Script

### Run Evaluation

```python
#!/usr/bin/env python3
"""
RAG Evaluation Harness

Usage:
    python scripts/evaluate_rag.py --test-pack technical-docs
    python scripts/evaluate_rag.py --test-pack personal-docs
    python scripts/evaluate_rag.py --all
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

def run_evaluation(test_pack_path: str, index_name: str):
    """
    Run evaluation test pack against RAG system.

    Args:
        test_pack_path: Path to JSON test pack
        index_name: PostgreSQL table name to query

    Returns:
        Evaluation results with scores
    """

    # Load test pack
    with open(test_pack_path) as f:
        test_pack = json.load(f)

    results = []

    for test_case in test_pack['test_cases']:
        print(f"\n{'='*70}")
        print(f"Test Case: {test_case['id']}")
        print(f"Query: {test_case['query']}")
        print(f"{'='*70}")

        # Run query
        start = time.time()
        answer, chunks = query_rag_system(test_case['query'], index_name)
        latency = time.time() - start

        # Score the answer
        scores = {
            'grounding': score_grounding(answer, chunks, test_case),
            'source_selection': score_sources(chunks, test_case),
            'completeness': score_completeness(answer, test_case),
            'no_invention': score_hallucination(answer, chunks),
            'multi_hop': score_multi_hop(answer, test_case) if test_case.get('multi_hop') else None,
            'latency': score_latency(latency)
        }

        # Calculate weighted total
        weights = test_pack['overall_scoring']
        total_score = sum(
            scores[k] * weights.get(f'{k}_weight', 0)
            for k in scores if scores[k] is not None
        )

        result = {
            'test_id': test_case['id'],
            'query': test_case['query'],
            'answer': answer,
            'latency_ms': latency * 1000,
            'chunks_retrieved': len(chunks),
            'scores': scores,
            'total_score': total_score,
            'passed': total_score >= 7.0  # 70% threshold
        }

        results.append(result)

        # Print summary
        print(f"\n📊 Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"   {metric}: {score:.1f}/10")
        print(f"   TOTAL: {total_score:.1f}/10 {'✅ PASS' if result['passed'] else '❌ FAIL'}")

    # Overall summary
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    avg_score = sum(r['total_score'] for r in results) / total

    print(f"\n{'='*70}")
    print(f"TEST PACK RESULTS: {test_pack['test_pack_name']}")
    print(f"{'='*70}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Average Score: {avg_score:.2f}/10")
    print(f"{'='*70}")

    return results
```

---

## Your Specific Test Packs

Based on your setup, I recommend:

### 1. **Technical-RAG-Pipeline** (Your own documentation)
- Test queries about your codebase
- Verify it understands GPU embeddings, RunPod, Railway
- Check multi-hop (e.g., "How does auto-detection work?")

### 2. **Personal-Documents** (PDFs, invoices, etc.)
- Entity extraction quality
- Date/metadata filtering
- Cross-document search

### 3. **Conversational-Search** (If you have chat logs)
- Person attribution
- Timeline queries
- Context preservation across messages

---

## Next Steps

Would you like me to:

1. **Create full test pack JSON files** for your specific use case?
2. **Build the evaluation script** to run these tests automatically?
3. **Generate sample test queries** based on your indexed documents?

Tell me which documents you want to evaluate against (your codebase? personal docs? chat logs?) and I'll create the appropriate test packs!