#!/usr/bin/env python3
"""
RAG Evaluation Harness

Systematic evaluation of RAG pipeline quality across multiple dimensions.

Usage:
    python scripts/evaluate_rag.py --index codebase_docs --test-pack eval/test_pack_technical_docs.json
    python scripts/evaluate_rag.py --index my_docs --test-pack eval/test_pack_personal_docs.json --output results.json
    python scripts/evaluate_rag.py --all  # Run all test packs

Output:
    - Console: Test results with scores
    - JSON: Detailed results with chunks, scores, timestamps
    - HTML: Visual dashboard with charts (optional)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import rag_low_level_m1_16gb_verbose as rag
from llama_index.core.schema import TextNode


# =============================================================================
# Scoring Functions
# =============================================================================

def score_grounding(answer: str, chunks: List[Dict], test_case: Dict) -> float:
    """
    Score how well the answer is grounded in retrieved chunks.

    Checks:
    - Are expected citations present?
    - Are claims verifiable in chunks?
    - No unsupported claims?

    Returns:
        Score 0-10
    """
    score = 10.0
    grading = test_case.get('grading_rules', {})
    expected_citations = test_case.get('expected_citations', [])

    # Check expected citations are in answer
    citations_found = sum(1 for cite in expected_citations if cite.lower() in answer.lower())
    citations_ratio = citations_found / len(expected_citations) if expected_citations else 1.0

    # Penalty for missing citations
    score *= citations_ratio

    # Check minimum citation count
    min_citations = grading.get('min_citations', 0)
    actual_citations = citations_found
    if actual_citations < min_citations:
        score -= (min_citations - actual_citations) * 1.5

    return max(0.0, min(10.0, score))


def score_sources(chunks: List[Dict], test_case: Dict) -> float:
    """
    Score source selection quality.

    Checks:
    - Were expected sources retrieved?
    - Precision/recall of source files
    - Relevance of chunks

    Returns:
        Score 0-10
    """
    expected_sources = test_case.get('expected_sources', [])

    if not expected_sources:
        return 10.0  # No specific sources required

    # Extract source files from chunks
    retrieved_sources = set()
    for chunk in chunks:
        source = chunk.get('metadata', {}).get('file_path', '')
        if source:
            retrieved_sources.add(source)

    # Calculate overlap
    expected_set = set(expected_sources)
    retrieved_set = retrieved_sources

    # Precision: how many retrieved are relevant
    if retrieved_set:
        precision = len(expected_set & retrieved_set) / len(retrieved_set)
    else:
        precision = 0.0

    # Recall: how many expected were retrieved
    if expected_set:
        recall = len(expected_set & retrieved_set) / len(expected_set)
    else:
        recall = 1.0

    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return f1 * 10.0


def score_completeness(answer: str, test_case: Dict) -> float:
    """
    Score answer completeness.

    Checks:
    - Must-mention topics covered?
    - Expected concepts present?
    - All required steps listed?

    Returns:
        Score 0-10
    """
    score = 10.0
    grading = test_case.get('grading_rules', {})

    # Check must_mention items
    must_mention = grading.get('must_mention', [])
    if must_mention:
        mentions_found = sum(1 for item in must_mention if item.lower() in answer.lower())
        mention_ratio = mentions_found / len(must_mention)
        score *= mention_ratio

    # Check expected concepts
    expected_concepts = grading.get('expected_concepts', [])
    if expected_concepts:
        concepts_found = sum(1 for concept in expected_concepts if any(word in answer.lower() for word in concept.lower().split()))
        concept_ratio = concepts_found / len(expected_concepts)
        score *= (0.5 + 0.5 * concept_ratio)  # 50% weight

    # Check expected steps
    expected_steps = grading.get('expected_steps', [])
    if expected_steps:
        steps_found = sum(1 for step in expected_steps if any(word in answer.lower() for word in step.lower().split()[:3]))
        step_ratio = steps_found / len(expected_steps)
        score *= (0.7 + 0.3 * step_ratio)  # 30% weight

    return max(0.0, min(10.0, score))


def score_hallucination(answer: str, chunks: List[Dict]) -> float:
    """
    Score hallucination resistance (no invention).

    Checks:
    - Claims verifiable in chunks?
    - No contradictions?
    - Appropriate "not found" when info missing?

    Returns:
        Score 0-10 (10 = no hallucination)
    """
    # Combine all chunk texts
    chunk_texts = [chunk.get('text', '') for chunk in chunks]
    combined_text = ' '.join(chunk_texts).lower()

    # Simple heuristic: check if answer contains info not in chunks
    # This is a basic check - more sophisticated NLI models could improve this

    answer_lower = answer.lower()

    # Positive signals (reduce hallucination risk)
    signals_good = [
        'according to' in answer_lower,
        'the document mentions' in answer_lower,
        'as stated in' in answer_lower,
        'not found' in answer_lower,
        'no information' in answer_lower
    ]

    # Negative signals (increase hallucination risk)
    signals_bad = [
        'i think' in answer_lower,
        'probably' in answer_lower,
        'might be' in answer_lower,
        'it seems' in answer_lower
    ]

    score = 10.0
    score += sum(signals_good) * 0.5
    score -= sum(signals_bad) * 2.0

    return max(0.0, min(10.0, score))


def score_multi_hop(answer: str, test_case: Dict) -> float:
    """
    Score multi-hop reasoning ability.

    Checks:
    - Combines info from multiple chunks/sources?
    - Maintains logical coherence?
    - Makes correct connections?

    Returns:
        Score 0-10
    """
    if not test_case.get('multi_hop'):
        return None  # Not a multi-hop question

    grading = test_case.get('grading_rules', {})

    # Check if answer shows synthesis
    synthesis_indicators = [
        'both' in answer.lower(),
        'compare' in answer.lower() or 'comparison' in answer.lower(),
        'while' in answer.lower(),
        'however' in answer.lower(),
        'whereas' in answer.lower(),
        'first' in answer.lower() and 'second' in answer.lower()
    ]

    score = 5.0  # Base score for multi-hop
    score += sum(synthesis_indicators) * 1.5

    return max(0.0, min(10.0, score))


def score_latency(latency_seconds: float) -> float:
    """
    Score query latency.

    Targets:
    - Excellent: < 2 seconds → 10 points
    - Good: 2-5 seconds → 7-9 points
    - Acceptable: 5-10 seconds → 4-6 points
    - Poor: > 10 seconds → 0-3 points

    Returns:
        Score 0-10
    """
    if latency_seconds < 2.0:
        return 10.0
    elif latency_seconds < 5.0:
        # Linear scale from 10 (2s) to 7 (5s)
        return 10.0 - (latency_seconds - 2.0) * 1.0
    elif latency_seconds < 10.0:
        # Linear scale from 7 (5s) to 4 (10s)
        return 7.0 - (latency_seconds - 5.0) * 0.6
    elif latency_seconds < 15.0:
        # Linear scale from 4 (10s) to 1 (15s)
        return 4.0 - (latency_seconds - 10.0) * 0.6
    else:
        return max(0.0, 1.0 - (latency_seconds - 15.0) * 0.1)


# =============================================================================
# RAG Query Function
# =============================================================================

def query_rag_system(query: str, index_name: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    """
    Query the RAG system and return answer + retrieved chunks.

    Args:
        query: Question to ask
        index_name: PostgreSQL table name
        top_k: Number of chunks to retrieve

    Returns:
        Tuple of (answer, list of chunk dicts with text and metadata)
    """
    # Configure settings
    rag.S.table = index_name
    rag.S.top_k = top_k

    # Build query engine
    embed_model = rag.build_embed_model()
    llm = rag.build_llm()
    vector_store = rag.make_vector_store()
    retriever = rag.VectorDBRetriever(vector_store, embed_model, top_k=top_k)

    from llama_index.core.query_engine import RetrieverQueryEngine
    query_engine = RetrieverQueryEngine(retriever=retriever, llm=llm)

    # Execute query
    response = query_engine.query(query)

    # Extract chunks
    chunks = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            chunks.append({
                'text': node.node.text,
                'metadata': node.node.metadata,
                'score': node.score
            })

    return str(response), chunks


# =============================================================================
# Main Evaluation Runner
# =============================================================================

def run_evaluation(test_pack_path: str, index_name: str, output_path: Optional[str] = None):
    """
    Run evaluation test pack against RAG system.

    Args:
        test_pack_path: Path to JSON test pack
        index_name: PostgreSQL table name to query
        output_path: Optional path to save results JSON

    Returns:
        Evaluation results dictionary
    """
    # Load test pack
    print(f"\n{'='*70}")
    print(f"Loading test pack: {test_pack_path}")
    print(f"{'='*70}\n")

    with open(test_pack_path) as f:
        test_pack = json.load(f)

    print(f"Test Pack: {test_pack['test_pack_name']}")
    print(f"Description: {test_pack['description']}")
    print(f"Test Cases: {len(test_pack['test_cases'])}")
    print()

    results = []
    weights = test_pack.get('overall_scoring', {})

    for i, test_case in enumerate(test_pack['test_cases'], 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}/{len(test_pack['test_cases'])}: {test_case['id']}")
        print(f"Query: {test_case['query']}")
        print(f"Difficulty: {test_case['difficulty']} | Multi-hop: {test_case.get('multi_hop', False)}")
        print(f"{'='*70}\n")

        # Run query
        start = time.time()
        try:
            answer, chunks = query_rag_system(test_case['query'], index_name, top_k=test_case.get('top_k', 5))
            latency = time.time() - start

            print(f"✅ Query completed in {latency:.2f}s")
            print(f"   Retrieved {len(chunks)} chunks")
            print(f"\n📝 Answer:\n{answer[:500]}{'...' if len(answer) > 500 else ''}\n")

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
            total_score = 0.0
            active_weights_sum = 0.0

            for metric, score in scores.items():
                if score is not None:
                    weight = weights.get(f'{metric}_weight', 0.0)
                    total_score += score * weight
                    active_weights_sum += weight

            # Normalize if weights don't sum to 1.0
            if active_weights_sum > 0:
                total_score = total_score / active_weights_sum * 10.0

            result = {
                'test_id': test_case['id'],
                'query': test_case['query'],
                'answer': answer,
                'latency_ms': latency * 1000,
                'chunks_retrieved': len(chunks),
                'chunks': chunks,
                'scores': {k: v for k, v in scores.items() if v is not None},
                'total_score': total_score,
                'passed': total_score >= test_pack.get('passing_threshold', 7.0),
                'timestamp': datetime.now().isoformat()
            }

            # Print scores
            print(f"📊 Scores:")
            for metric, score in scores.items():
                if score is not None:
                    emoji = "✅" if score >= 7.0 else "⚠️" if score >= 4.0 else "❌"
                    print(f"   {emoji} {metric:20s}: {score:4.1f}/10")

            emoji_total = "✅" if result['passed'] else "❌"
            print(f"\n   {emoji_total} TOTAL: {total_score:.1f}/10")

            if not result['passed']:
                print(f"\n   ⚠️  FAILED - Score below threshold ({test_pack.get('passing_threshold', 7.0)})")

        except Exception as e:
            print(f"❌ Query failed: {e}")
            result = {
                'test_id': test_case['id'],
                'query': test_case['query'],
                'error': str(e),
                'passed': False,
                'total_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }

        results.append(result)

    # Overall summary
    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)
    avg_score = sum(r.get('total_score', 0.0) for r in results) / total if total > 0 else 0.0
    avg_latency = sum(r.get('latency_ms', 0.0) for r in results if 'latency_ms' in r) / sum(1 for r in results if 'latency_ms' in r)

    print(f"\n\n{'='*70}")
    print(f"TEST PACK SUMMARY: {test_pack['test_pack_name']}")
    print(f"{'='*70}")
    print(f"✅ Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"📊 Average Score: {avg_score:.2f}/10")
    print(f"⏱️  Average Latency: {avg_latency:.0f}ms")
    print(f"{'='*70}\n")

    # Save results
    if output_path:
        output_data = {
            'test_pack': test_pack['test_pack_name'],
            'timestamp': datetime.now().isoformat(),
            'index_name': index_name,
            'summary': {
                'total_tests': total,
                'passed': passed,
                'failed': total - passed,
                'pass_rate': passed / total if total > 0 else 0.0,
                'average_score': avg_score,
                'average_latency_ms': avg_latency
            },
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Results saved to: {output_path}\n")

    return {
        'summary': {
            'passed': passed,
            'total': total,
            'avg_score': avg_score,
            'avg_latency_ms': avg_latency
        },
        'results': results
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Harness")

    parser.add_argument(
        '--index',
        required=True,
        help='PostgreSQL table name to query (e.g., codebase_docs)'
    )

    parser.add_argument(
        '--test-pack',
        required=True,
        help='Path to test pack JSON file'
    )

    parser.add_argument(
        '--output',
        help='Path to save results JSON (optional)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of chunks to retrieve (default: 5)'
    )

    args = parser.parse_args()

    # Validate inputs
    test_pack_path = Path(args.test_pack)
    if not test_pack_path.exists():
        print(f"❌ Error: Test pack not found: {test_pack_path}")
        sys.exit(1)

    # Run evaluation
    try:
        results = run_evaluation(
            test_pack_path=str(test_pack_path),
            index_name=args.index,
            output_path=args.output
        )

        # Exit with appropriate code
        if results['summary']['passed'] < results['summary']['total']:
            sys.exit(1)  # Some tests failed
        else:
            sys.exit(0)  # All tests passed

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
