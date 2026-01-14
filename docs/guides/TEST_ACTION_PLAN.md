# Test Coverage Action Plan
**Prioritized Implementation Roadmap**

**Goal:** Increase test coverage from 30% to 80% over 3 phases
**Timeline:** 8 weeks
**Estimated Effort:** ~80 hours

---

## Phase 1: Critical Risk Mitigation (Week 1-2)
**Target Coverage:** 30% → 50% (+20 points)
**Effort:** ~27 hours

### Priority 0: Must-Have Tests (Week 1)

#### Task 1.1: MLX Embedding Tests (4 hours)
**File:** `tests/test_mlx_embedding.py`
**Status:** NEW FILE
**Risk:** HIGH - M1 users get no acceleration if broken

```bash
# Create test file
touch tests/test_mlx_embedding.py

# Copy template from TEST_IMPLEMENTATION_GUIDE.md (Template 3)
# Add these test classes:
- TestMLXEmbeddingInit (3 tests)
- TestMLXEmbeddingQuery (4 tests)
- TestMLXEmbeddingBatch (3 tests)
- TestMLXPerformance (2 tests)
- TestMLXMemoryUsage (1 test)

# Run tests
pytest tests/test_mlx_embedding.py -v
```

**Acceptance Criteria:**
- [ ] 13+ tests added
- [ ] Tests pass on M1 Mac
- [ ] Tests skip gracefully on non-M1
- [ ] Coverage for mlx_embedding.py > 80%

---

#### Task 1.2: Reranker Tests (3 hours)
**File:** `tests/test_reranker.py`
**Status:** NEW FILE
**Risk:** HIGH - Retrieval quality degrades without reranking

```bash
# Create test file
touch tests/test_reranker.py

# Add test classes:
- TestCrossEncoderReranker (5 tests)
- TestMMRReranker (4 tests)
- TestScoreNormalization (3 tests)
- TestRerankerEdgeCases (3 tests)

# Run tests
pytest tests/test_reranker.py -v
```

**Tests to Add:**
```python
def test_cross_encoder_init()
def test_cross_encoder_rerank_basic()
def test_cross_encoder_score_range()
def test_mmr_diversity_scoring()
def test_mmr_lambda_parameter()
def test_empty_results_handling()
def test_single_result_reranking()
```

**Acceptance Criteria:**
- [ ] 15+ tests added
- [ ] Coverage for reranker.py > 75%
- [ ] All reranking strategies tested
- [ ] Edge cases covered

---

#### Task 1.3: HyDE Retrieval Tests (3 hours)
**File:** `tests/test_hyde_retrieval.py`
**Status:** NEW FILE
**Risk:** HIGH - Advanced retrieval fails silently

```bash
# Create test file
touch tests/test_hyde_retrieval.py

# Add test classes:
- TestHyDEGeneration (4 tests)
- TestHyDERetrieval (5 tests)
- TestHyDEConfiguration (3 tests)
```

**Key Tests:**
- Hypothetical document generation
- Document expansion with LLM
- Multi-document synthesis
- Fallback to regular retrieval
- Error handling

**Acceptance Criteria:**
- [ ] 12+ tests added
- [ ] Coverage for hyde_retrieval.py > 70%
- [ ] LLM integration mocked properly
- [ ] Fallback behavior tested

---

#### Task 1.4: Database Integration Tests (6 hours)
**File:** `tests/test_database_real_integration.py`
**Status:** NEW FILE (existing test_database_integration.py is minimal)
**Risk:** HIGH - Production DB operations untested

```bash
# Create comprehensive integration test file
touch tests/test_database_real_integration.py

# Add test classes:
- TestDatabaseConnection (5 tests)
- TestVectorOperations (8 tests)
- TestTransactionHandling (4 tests)
- TestErrorRecovery (5 tests)
- TestConcurrency (3 tests)
```

**Setup Requirements:**
```yaml
# .github/workflows/ci.yml addition
- name: Start PostgreSQL
  run: |
    docker run -d -p 5432:5432 \
      -e POSTGRES_PASSWORD=test \
      -e POSTGRES_DB=test_db \
      ankane/pgvector
    sleep 5
```

**Key Tests:**
```python
def test_connection_pool_exhaustion()
def test_vector_similarity_search_real()
def test_transaction_rollback()
def test_concurrent_writes()
def test_index_creation()
def test_connection_recovery()
```

**Acceptance Criteria:**
- [ ] 25+ integration tests added
- [ ] Tests run in CI with Docker PostgreSQL
- [ ] Connection pool tested
- [ ] Transaction handling verified
- [ ] Concurrent access tested

---

#### Task 1.5: Conversation Memory Tests (3 hours)
**File:** `tests/test_conversation_memory.py`
**Status:** NEW FILE
**Risk:** MEDIUM - Chat functionality fails

```bash
touch tests/test_conversation_memory.py

# Add test classes:
- TestConversationMemoryInit (3 tests)
- TestMessageStorage (5 tests)
- TestContextPruning (4 tests)
- TestMemoryRetrieval (4 tests)
```

**Key Tests:**
- Message storage and retrieval
- Conversation history management
- Context window pruning
- Multi-turn conversation
- Memory persistence

**Acceptance Criteria:**
- [ ] 16+ tests added
- [ ] Coverage for conversation_memory.py > 75%
- [ ] Chat flow tested end-to-end
- [ ] Pruning logic verified

---

#### Task 1.6: Query Cache Tests (2 hours)
**File:** `tests/test_query_cache.py`
**Status:** NEW FILE
**Risk:** MEDIUM - Performance degradation

Use Template 2 from TEST_IMPLEMENTATION_GUIDE.md

**Acceptance Criteria:**
- [ ] 15+ tests added
- [ ] Coverage for query_cache.py > 80%
- [ ] TTL expiration tested
- [ ] Cache eviction verified
- [ ] Hit/miss metrics tested

---

#### Task 1.7: Query Expansion Tests (2 hours)
**File:** `tests/test_query_expansion.py`
**Status:** NEW FILE
**Risk:** HIGH - Query quality degrades

```python
# Key tests to add:
def test_expand_query_with_synonyms()
def test_extract_keywords()
def test_query_rewriting()
def test_multi_query_generation()
```

**Acceptance Criteria:**
- [ ] 12+ tests added
- [ ] Coverage for query_expansion.py > 70%
- [ ] Synonym generation tested
- [ ] Query rewriting verified

---

#### Task 1.8: Parent-Child Chunking Tests (2 hours)
**File:** `tests/test_parent_child_chunking.py`
**Status:** NEW FILE
**Risk:** MEDIUM - Advanced chunking fails

**Acceptance Criteria:**
- [ ] 10+ tests added
- [ ] Coverage for parent_child_chunking.py > 70%
- [ ] Hierarchy preserved
- [ ] Chunk relationships tested

---

#### Task 1.9: Metrics Collection Tests (2 hours)
**File:** `tests/test_metrics.py`
**Status:** NEW FILE
**Risk:** MEDIUM - Monitoring broken

**Acceptance Criteria:**
- [ ] 10+ tests added
- [ ] Coverage for metrics.py > 65%
- [ ] Metric recording verified
- [ ] Aggregation tested

---

### Week 1 Deliverables Checklist

- [ ] 9 new test files created
- [ ] 130+ new tests added
- [ ] All high-risk utils covered
- [ ] Coverage: 30% → 45%
- [ ] CI passing with new tests

---

## Phase 2: Production Readiness (Week 3-4)
**Target Coverage:** 50% → 65% (+15 points)
**Effort:** ~29 hours

### Priority 1: Application Layer Tests (Week 3)

#### Task 2.1: Web Backend API Tests (8 hours)
**File:** `tests/test_web_backend_api.py`
**Status:** NEW FILE
**Risk:** HIGH - Production API untested

```bash
touch tests/test_web_backend_api.py

# Add test classes:
- TestAPIHealth (3 tests)
- TestQueryEndpoint (10 tests)
- TestUploadEndpoint (8 tests)
- TestErrorHandling (6 tests)
- TestAuthentication (4 tests)
- TestRateLimiting (3 tests)
```

**Setup:**
```python
from fastapi.testclient import TestClient
from rag_web_backend import app

@pytest.fixture
def client():
    return TestClient(app)
```

**Key Tests:**
- Query endpoint validation
- File upload handling
- Error responses (4xx, 5xx)
- Authentication
- Rate limiting
- Response streaming

**Acceptance Criteria:**
- [ ] 34+ API tests added
- [ ] Coverage for rag_web_backend.py > 60%
- [ ] All endpoints tested
- [ ] Error handling verified
- [ ] Integration with query engine tested

---

#### Task 2.2: Web UI Tests (Streamlit) (4 hours)
**File:** `tests/test_web_ui.py`
**Status:** NEW FILE
**Risk:** MEDIUM - UI completely untested

```bash
touch tests/test_web_ui.py

# Note: Streamlit testing is tricky, focus on business logic
# Use AppTest from streamlit.testing.v1

from streamlit.testing.v1 import AppTest

def test_web_ui_loads():
    at = AppTest.from_file("rag_web.py")
    at.run()
    assert not at.exception
```

**Acceptance Criteria:**
- [ ] 10+ UI tests added
- [ ] Page loads without errors
- [ ] Query submission works
- [ ] File upload works
- [ ] Settings persistence tested

---

#### Task 2.3: vLLM Integration Tests (4 hours)
**File:** `tests/test_vllm_integration.py`
**Status:** NEW FILE
**Risk:** HIGH - vLLM completely untested

```bash
touch tests/test_vllm_integration.py

# Add test classes:
- TestVLLMClient (6 tests)
- TestVLLMWrapper (5 tests)
- TestVLLMErrorHandling (4 tests)
```

**Key Tests:**
- Client initialization
- Request formatting
- Response parsing
- Timeout handling
- Fallback to llama.cpp

**Acceptance Criteria:**
- [ ] 15+ tests added
- [ ] Coverage for vllm_client.py > 70%
- [ ] Coverage for vllm_wrapper.py > 70%
- [ ] Mock vLLM server for testing

---

#### Task 2.4: Full E2E Pipeline Tests (6 hours)
**File:** `tests/test_e2e_full_pipeline.py` (extend existing)
**Status:** EXTEND EXISTING
**Risk:** HIGH - End-to-end flow untested

```bash
# Extend existing test_e2e_pipeline.py

# Add test classes:
- TestFullIndexingPipeline (5 tests)
- TestFullQueryPipeline (6 tests)
- TestMultiFormatPipeline (4 tests)
- TestLargeDatasetPipeline (3 tests)
```

**Key Tests:**
```python
def test_full_pdf_indexing_and_query()
def test_full_html_indexing_and_query()
def test_multi_document_indexing()
def test_query_with_reranking()
def test_query_with_hyde()
def test_chat_conversation_flow()
```

**Acceptance Criteria:**
- [ ] 18+ E2E tests added
- [ ] Real documents indexed
- [ ] Real database used
- [ ] Full pipeline tested
- [ ] Performance measured

---

#### Task 2.5: Error Handling & Edge Cases (4 hours)
**File:** `tests/test_error_handling.py`
**Status:** NEW FILE
**Risk:** HIGH - Error scenarios untested

```bash
touch tests/test_error_handling.py

# Add test classes:
- TestDatabaseErrors (6 tests)
- TestLLMErrors (5 tests)
- TestEmbeddingErrors (4 tests)
- TestNetworkErrors (4 tests)
```

**Key Error Scenarios:**
```python
def test_database_connection_lost()
def test_llm_generation_timeout()
def test_embedding_model_not_found()
def test_disk_full_during_indexing()
def test_cuda_out_of_memory()
def test_invalid_pdf_file()
def test_network_timeout()
```

**Acceptance Criteria:**
- [ ] 19+ error tests added
- [ ] All critical error paths tested
- [ ] Graceful degradation verified
- [ ] Error messages user-friendly

---

#### Task 2.6: RAG Benchmark Tests (3 hours)
**File:** `tests/test_rag_benchmark.py`
**Status:** NEW FILE
**Risk:** MEDIUM - Benchmarking unreliable

**Acceptance Criteria:**
- [ ] 8+ benchmark tests added
- [ ] Coverage for rag_benchmark.py > 60%
- [ ] Benchmarks reproducible

---

### Week 3-4 Deliverables Checklist

- [ ] 6 new/extended test files
- [ ] 100+ new tests added
- [ ] Web backend fully tested
- [ ] vLLM integration tested
- [ ] E2E tests comprehensive
- [ ] Coverage: 50% → 65%
- [ ] CI passing

---

## Phase 3: Quality Excellence (Week 5-8)
**Target Coverage:** 65% → 80% (+15 points)
**Effort:** ~24 hours

### Priority 2: Advanced Testing (Week 5-6)

#### Task 3.1: Chaos/Resilience Testing (16 hours)
**File:** `tests/chaos/test_resilience.py`
**Status:** NEW FILE
**Risk:** LOW - Nice to have

```bash
mkdir tests/chaos
touch tests/chaos/test_resilience.py

# Add chaos scenarios:
- Database goes down during query
- Disk fills up during indexing
- Network partition
- Memory pressure
- CPU throttling
```

**Tools:**
```bash
pip install chaos-engineering
```

**Acceptance Criteria:**
- [ ] 15+ chaos tests added
- [ ] System resilience verified
- [ ] Graceful degradation tested
- [ ] Recovery mechanisms tested

---

#### Task 3.2: Load Testing (12 hours)
**File:** `tests/performance/test_load.py`
**Status:** NEW FILE
**Risk:** LOW - Production readiness

```bash
mkdir tests/performance
touch tests/performance/test_load.py

# Use locust or pytest-benchmark
pip install locust
```

**Load Scenarios:**
- 10 concurrent users
- 100 concurrent users
- 1000 queries/minute
- Large file uploads
- Sustained load (1 hour)

**Acceptance Criteria:**
- [ ] 10+ load tests added
- [ ] Performance baselines established
- [ ] Bottlenecks identified
- [ ] Scalability measured

---

#### Task 3.3: Mutation Testing (8 hours)
**File:** N/A (mutmut configuration)
**Status:** NEW TOOL
**Risk:** LOW - Quality assurance

```bash
pip install mutmut

# Configure mutmut
cat > .mutmut.toml << EOF
[mutmut]
paths_to_mutate=utils/,rag_low_level_m1_16gb_verbose.py
tests_dir=tests/
runner=pytest
EOF

# Run mutation tests
mutmut run
mutmut results
```

**Acceptance Criteria:**
- [ ] Mutation testing configured
- [ ] Baseline mutation score established
- [ ] Weak tests identified
- [ ] Tests strengthened

---

#### Task 3.4: Contract Testing (4 hours)
**File:** `tests/contract/test_api_contract.py`
**Status:** NEW FILE
**Risk:** LOW - API compatibility

```bash
mkdir tests/contract
touch tests/contract/test_api_contract.py

# Use pact for contract testing
pip install pact-python
```

**Acceptance Criteria:**
- [ ] API contracts defined
- [ ] Contract tests passing
- [ ] Breaking changes detected

---

### Priority 3: Continuous Improvements (Week 7-8)

#### Task 3.5: Test Documentation (8 hours)

**Create:**
- [ ] TESTING_GUIDE.md (comprehensive guide)
- [ ] Test writing standards
- [ ] Example test patterns
- [ ] Troubleshooting guide
- [ ] Best practices checklist

---

#### Task 3.6: CI/CD Enhancements (8 hours)

**Improvements:**
- [ ] Parallel test execution in CI
- [ ] Test result analytics
- [ ] Flaky test detection
- [ ] Performance trend tracking
- [ ] Automated coverage reports

---

#### Task 3.7: Test Maintenance (8 hours)

**Tasks:**
- [ ] Remove duplicate tests
- [ ] Consolidate fixtures
- [ ] Optimize slow tests
- [ ] Update documentation
- [ ] Review test patterns

---

## Weekly Progress Tracking

### Week 1
- [ ] MLX embedding tests (4h)
- [ ] Reranker tests (3h)
- [ ] HyDE retrieval tests (3h)
- [ ] Database integration tests (6h)
- [ ] Conversation memory tests (3h)
- [ ] Query cache tests (2h)
- [ ] Query expansion tests (2h)
- [ ] Parent-child chunking tests (2h)
- [ ] Metrics tests (2h)
**Target:** 27 hours, Coverage 30% → 45%

### Week 2
- [ ] Review and fix failing tests
- [ ] Improve coverage gaps
- [ ] Add missing edge cases
**Target:** Coverage 45% → 50%

### Week 3-4
- [ ] Web backend API tests (8h)
- [ ] Web UI tests (4h)
- [ ] vLLM integration tests (4h)
- [ ] Full E2E tests (6h)
- [ ] Error handling tests (4h)
- [ ] Benchmark tests (3h)
**Target:** 29 hours, Coverage 50% → 65%

### Week 5-6
- [ ] Chaos testing (16h)
- [ ] Load testing (12h)
**Target:** 28 hours, Coverage 65% → 75%

### Week 7-8
- [ ] Mutation testing (8h)
- [ ] Contract testing (4h)
- [ ] Documentation (8h)
- [ ] CI/CD enhancements (8h)
- [ ] Test maintenance (8h)
**Target:** 36 hours, Coverage 75% → 80%

---

## Success Metrics

### Coverage Milestones
- [x] Week 0: 30% (baseline)
- [ ] Week 2: 50% (+20 points)
- [ ] Week 4: 65% (+15 points)
- [ ] Week 8: 80% (+15 points)

### Quality Metrics
- [ ] Zero failing tests in CI
- [ ] Test execution < 5 minutes
- [ ] Flaky test rate < 1%
- [ ] Mutation score > 70%
- [ ] All critical paths tested

### Documentation
- [ ] Test guide complete
- [ ] All tests documented
- [ ] Examples provided
- [ ] Troubleshooting guide ready

---

## Risk Mitigation

### If Behind Schedule

**Priority Cuts:**
1. Skip mutation testing (Week 7)
2. Skip contract testing (Week 8)
3. Reduce chaos testing scenarios
4. Defer load testing to future sprint

**Must-Have Tests:**
- All utility modules (Week 1)
- Web backend API (Week 3)
- E2E pipeline (Week 4)

---

## Getting Started

### Day 1: Quick Start
```bash
# 1. Create test file structure
mkdir -p tests/chaos tests/performance tests/contract

# 2. Start with highest priority
cd tests
touch test_mlx_embedding.py

# 3. Copy template from TEST_IMPLEMENTATION_GUIDE.md
# 4. Write first test
# 5. Run it
pytest tests/test_mlx_embedding.py -v

# 6. Check coverage
pytest tests/test_mlx_embedding.py --cov=utils.mlx_embedding --cov-report=term-missing

# 7. Iterate until coverage > 80%
```

---

## Daily Checklist

**Every Day:**
- [ ] Write tests for 1-2 hours
- [ ] Run `pytest tests/ -v`
- [ ] Check coverage: `pytest --cov=. --cov-report=term-missing`
- [ ] Commit passing tests
- [ ] Update progress in this file

**Every Week:**
- [ ] Review progress vs. plan
- [ ] Update coverage metrics
- [ ] Adjust priorities if needed
- [ ] Document any blockers

---

## Contact & Support

**Questions?**
- Review: TESTING_QUALITY_AUDIT.md (comprehensive analysis)
- Review: TEST_IMPLEMENTATION_GUIDE.md (templates & examples)
- Check: pytest documentation
- Check: existing test files for patterns

**Stuck?**
- Look at similar test files
- Use fixtures from conftest.py
- Mock external dependencies
- Start simple, add complexity

---

**Last Updated:** January 9, 2026
**Status:** Phase 1 - Week 1 starting
**Current Coverage:** 30%
**Target Coverage:** 80%
