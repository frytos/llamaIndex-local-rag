# Production-Ready RAG System

## What This Is

A web-accessible, secure RAG (Retrieval-Augmented Generation) pipeline that transforms a local document indexing system into a production-ready service with authentication, automated deployment, and hybrid cloud architecture. Built for personal use with enterprise-grade security and privacy controls.

## Core Value

Private document intelligence accessible from anywhere with zero-trust security — your documents never leave your control, yet you can query them from any browser with proper authentication.

## Requirements

### Validated

<!-- Shipped capabilities from existing codebase -->

- ✓ Local RAG pipeline with document indexing (PDF, HTML, DOCX, code files) — existing
- ✓ PostgreSQL + pgvector vector storage with HNSW indices (100x+ faster queries) — existing
- ✓ HuggingFace embedding models (bge-small-en, all-MiniLM-L6-v2, bge-m3) with MLX acceleration — existing
- ✓ LLM integration (llama.cpp GGUF, vLLM AWQ) for Mistral 7B — existing
- ✓ Streamlit web UI for local querying — existing
- ✓ CLI interface (`rag_interactive.py`) for pipeline operations — existing
- ✓ RunPod deployment scripts for GPU workloads — existing
- ✓ Performance monitoring (Prometheus metrics, query logging) — existing
- ✓ RAG enhancements (query expansion, reranking, semantic caching) — existing
- ✓ Apple Silicon (MPS) and NVIDIA (CUDA) GPU support — existing
- ✓ Conversation memory and context management — existing

### Active

<!-- Building toward these -->

- [ ] Web-accessible interface (accessible from any browser, not just localhost)
- [ ] Username/password authentication system
- [ ] Rate limiting for API protection
- [ ] Data privacy controls (user documents isolated and encrypted)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Hybrid deployment architecture (Vercel/Railway frontend + RunPod backend)
- [ ] Production-ready security (HTTPS, secrets management, secure session handling)
- [ ] Automated deployment on git push

### Out of Scope

- Multi-tenancy support — Single user system, no need for tenant isolation or complex permission models
- OAuth/social login — Username/password sufficient for personal use
- Advanced model fine-tuning — Use existing pre-trained models

## Context

**Current State:**
- Comprehensive local RAG pipeline optimized for M1 Mac (16GB) and GPU servers (RTX 4090)
- All operations currently require local Python environment setup
- No authentication or access controls
- RunPod deployment scripts exist but require manual execution
- Streamlit UI runs on localhost only

**User Context:**
- Personal RAG system for private document querying
- Need to access system remotely without local setup
- Privacy-critical: documents contain sensitive personal information
- Single user (no team collaboration needed)

**Technical Environment:**
- Python 3.11+ with LlamaIndex framework
- PostgreSQL 16 with pgvector extension
- 80+ utility scripts for operations and benchmarking
- Existing monitoring with Prometheus and Grafana
- Comprehensive documentation in `docs/` directory

## Constraints

- **Privacy**: Hard requirement — documents must never be accessible to third parties or cloud providers without explicit encryption
- **Architecture**: Hybrid deployment required — GPU-intensive tasks (embedding, LLM) must run on RunPod, web interface on serverless platform
- **Cost**: No multi-tenancy overhead — optimize for single-user deployment

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hybrid architecture (Vercel/Railway + RunPod) | Separate web interface from GPU workloads for cost optimization and performance | — Pending |
| Username/password auth (not OAuth) | Personal system doesn't need social login complexity | — Pending |
| GitHub Actions for CI/CD | Standard platform with good GitHub integration | — Pending |
| Keep existing RunPod backend | Proven GPU infrastructure, no need to migrate | — Pending |

---
*Last updated: 2026-01-15 after initialization*
