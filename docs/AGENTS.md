# Repository Guidelines

## Project Structure & Module Organization
Root scripts `rag_minimal_local.py` and `rag_low_level_m1_16gb_verbose.py` hold the disk-based and Postgres RAG flows. Put PDFs and DOCX files under `data/`, keep `db-init/001-pgvector.sql` and `docker-compose.yml` aligned with the pgvector database, and stash helper utilities in `scripts/`. Generated stores, GGUF weights, and experimental artifacts belong in ignored scratch directories.

## Build, Test, and Development Commands
- `python rag_minimal_local.py index --docs data/llama2.pdf data/facebook-groussarda-14_08_2024-c4NOel1H --store-dir .store --embedding-model sentence-transformers/all-MiniLM-L6-v2` – ingest, chunk, and embed documents into `.store`.
- `python rag_minimal_local.py ask --store-dir .store --question "How is llama2 trained?" --k 4 --llm-model-path ~/Models/mistral.gguf` – retrieve the top chunks and answer with llama.cpp when provided.
- `docker compose up db` – run the pgvector Postgres instance with the init script; use `docker compose logs db` for health checks.
- `python rag_low_level_m1_16gb_verbose.py` – execute the verbose Postgres pipeline; override env vars (e.g., `EMBED_MODEL`, `RESET_TABLE`, `QUESTION`) instead of editing code.

## Coding Style & Naming Conventions
Target Python 3.11+, follow PEP 8 with 4-space indents and `snake_case` helpers, and mirror the existing module sections (parsers, chunking, embeddings, storage, commands). Public functions should be type-annotated; reusable configuration belongs in dataclasses (`StoreConfig`, `Settings`) rather than loose dicts. Keep new CLI switches inside the current `argparse` subparsers and narrate only tricky logic with brief comments.

## Testing Guidelines
No automated tests exist, so rely on manual checks. After `index`, verify `.store/config.json`, `.store/chunks.jsonl`, and `.store/embeddings.npy` plus a quick spot-check of chunk metadata. For the Postgres path, wait until `docker compose ps` marks the database healthy, then inspect logged chunk counts, embedding batches, and retrieved nodes; rerun the same prompt with fixed `--k` to compare results across changes.

## Commit & Pull Request Guidelines
Git history uses short imperative subjects (`adds .gitignore for project`), so keep titles direct and reserve body text for rationale or rollback notes. Pull requests must describe scope, affected commands, new env vars, and whether contributors need to rebuild stores or reset volumes. Include reproduction steps (exact CLI or Docker commands) plus relevant log snippets or screenshots before requesting review.

## Security & Configuration Tips
Default credentials (`fryt` / `frytos`) are for local development only; override them with env vars or a `.env` file before sharing commands or logs. Never commit downloaded GGUF weights, generated stores, or raw documents, and surface future secrets through `os.getenv` fallbacks documented in this guide.
