# GUI Feature Parity Plan - CLI to GUI Migration

**Date:** 2026-01-08
**Goal:** Implement ALL CLI/terminal options in the Streamlit GUI
**Status:** Analysis Complete - Implementation Plan Ready

---

## 📊 Current State Analysis

### ✅ Already Implemented in GUI

| Feature | CLI Method | GUI Location | Status |
|---------|-----------|--------------|--------|
| Document selection | `PDF_PATH=path` | Quick Start / Advanced Index | ✅ Complete |
| Chunk size | `CHUNK_SIZE=700` | Quick Start / Advanced Index | ✅ Complete |
| Chunk overlap | `CHUNK_OVERLAP=150` | Quick Start / Advanced Index | ✅ Complete |
| Embedding model | `EMBED_MODEL=model` | Advanced Index | ✅ Complete |
| Embedding dimensions | `EMBED_DIM=384` | Advanced Index (derived) | ✅ Complete |
| Embedding batch | `EMBED_BATCH=64` | Quick Start / Advanced Index | ✅ Complete |
| Embedding backend | `EMBED_BACKEND=mlx` | Advanced Index | ✅ Complete |
| Table name | `PGTABLE=name` | Quick Start / Advanced Index | ✅ Complete |
| Reset table | `RESET_TABLE=1` | Quick Start / Advanced Index | ✅ Complete |
| TOP_K | `TOP_K=4` | Query page | ✅ Complete |
| Temperature | `TEMP=0.1` | Query page | ✅ Complete |
| Max tokens | `MAX_NEW_TOKENS=256` | Query page | ✅ Complete |
| Context window | `CTX=3072` | Query page | ✅ Complete |
| Query expansion | `ENABLE_QUERY_EXPANSION=1` | Query page (Advanced Features) | ✅ Complete |
| Reranking | `ENABLE_RERANKING=1` | Query page (Advanced Features) | ✅ Complete |
| Hybrid search | `HYBRID_ALPHA=0.7` | Query page (Advanced Features) | ✅ Complete |
| Metadata filtering | `ENABLE_FILTERS=1` | Query page (Advanced Features) | ✅ Complete |

### ❌ Missing from GUI (Need Implementation)

| Feature | CLI Method | Priority | Complexity |
|---------|-----------|----------|------------|
| **Index-only mode** | `--index-only` | High | Low |
| **Query-only mode** | `--query-only` | High | Low |
| **Interactive REPL** | `--interactive` | Medium | Medium |
| **Skip validation** | `--skip-validation` | Low | Low |
| **Reset database** | `RESET_DB=1` | Low | Low |
| **Chat metadata extraction** | `EXTRACT_CHAT_METADATA=1` | Medium | Low |
| **MMR diversity** | `MMR_THRESHOLD=0.5` | Medium | Low |
| **Query expansion method** | `QUERY_EXPANSION_METHOD=llm` | Medium | Low |
| **Query expansion count** | `QUERY_EXPANSION_COUNT=2` | Medium | Low |
| **HyDE retrieval** | `ENABLE_HYDE=1` | Low | Medium |
| **HyDE hypotheses** | `HYDE_NUM_HYPOTHESES=2` | Low | Low |
| **HyDE hypothesis length** | `HYDE_HYPOTHESIS_LENGTH=100` | Low | Low |
| **HyDE fusion method** | `HYDE_FUSION_METHOD=rrf` | Low | Low |
| **Rerank candidates** | `RERANK_CANDIDATES=12` | Medium | Low |
| **Rerank TOP_K** | `RERANK_TOP_K=4` | Medium | Low |
| **Rerank model** | `RERANK_MODEL=model` | Low | Low |
| **Semantic cache** | `ENABLE_SEMANTIC_CACHE=1` | High | Low |
| **Cache threshold** | `SEMANTIC_CACHE_THRESHOLD=0.92` | Medium | Low |
| **Cache max size** | `SEMANTIC_CACHE_MAX_SIZE=1000` | Low | Low |
| **Cache TTL** | `SEMANTIC_CACHE_TTL=86400` | Low | Low |
| **Conversation memory** | `ENABLE_CONVERSATION_MEMORY=1` | High | High |
| **Max conversation turns** | `MAX_CONVERSATION_TURNS=10` | Medium | Low |
| **Conversation timeout** | `CONVERSATION_TIMEOUT=3600` | Low | Low |
| **Auto summarize** | `AUTO_SUMMARIZE=1` | Medium | Medium |
| **Summarize threshold** | `SUMMARIZE_THRESHOLD=5` | Low | Low |
| **LLM model URL** | `MODEL_URL=url` | Medium | Medium |
| **LLM model path** | `MODEL_PATH=path` | Low | Medium |
| **GPU layers** | `N_GPU_LAYERS=24` | Low | Low |
| **Batch size** | `N_BATCH=256` | Low | Low |
| **vLLM backend** | Use vLLM instead of llama.cpp | Low | High |
| **Single query CLI** | `--query "question"` | Medium | Medium |
| **Performance logging** | Built-in performance tracking | High | Low |
| **Batch indexing** | Index multiple docs at once | Medium | Medium |
| **Export results** | Save query results | Medium | Low |

---

## 🎯 Implementation Plan by Agent

### 1. **Product Manager** - Feature Prioritization & Requirements

**Responsibilities:**
- Define user stories for each missing feature
- Prioritize features by user impact
- Create acceptance criteria
- Define success metrics

**Tasks:**

#### Priority 1: Core Missing Features (Week 1)
```
USER STORIES:

US-001: Index-Only Mode
- As a user, I want to index documents without querying
- So that I can batch index multiple documents efficiently
- Acceptance: Checkbox "Index only (skip query)" on Quick Start

US-002: Query-Only Mode
- As a user, I want to query without re-indexing
- So that I can quickly search existing indexes
- Acceptance: Disabled "Re-index" checkbox on Query page

US-003: Semantic Cache Controls
- As a user, I want to control query caching
- So that I can optimize for speed vs freshness
- Acceptance: Cache settings in Query page (threshold, max size, TTL)

US-004: Performance Dashboard
- As a user, I want to see indexing/query performance metrics
- So that I can optimize my configuration
- Acceptance: Performance tab showing timing breakdown
```

#### Priority 2: Advanced Retrieval (Week 2)
```
US-005: MMR Diversity Control
- As a user, I want to control result diversity
- So that I don't get repetitive chunks
- Acceptance: MMR slider in Advanced Features (0.0-1.0)

US-006: Query Expansion Configuration
- As a user, I want to choose expansion method and count
- So that I can balance quality vs speed
- Acceptance: Expansion method dropdown + count slider

US-007: Reranking Fine-Tuning
- As a user, I want to configure reranking parameters
- So that I can optimize retrieval quality
- Acceptance: Rerank candidates, TOP_K, model selection

US-008: Chat Metadata Extraction
- As a user, I want to control metadata extraction
- So that I can enable/disable participant/date parsing
- Acceptance: Checkbox "Extract chat metadata" in indexing
```

#### Priority 3: Advanced Features (Week 3)
```
US-009: HyDE Retrieval
- As a power user, I want HyDE retrieval options
- So that I can improve technical query results
- Acceptance: HyDE section in Advanced Features

US-010: Conversation Memory
- As a user, I want multi-turn conversations
- So that I can ask follow-up questions naturally
- Acceptance: Conversation mode with history and context

US-011: Interactive REPL Mode
- As a developer, I want REPL-style querying in GUI
- So that I can quickly iterate on queries
- Acceptance: Chat-style interface with command history

US-012: LLM Model Configuration
- As a user, I want to configure LLM model and parameters
- So that I can use different models or local files
- Acceptance: Settings page with model URL/path, GPU layers
```

**Deliverables:**
- ✅ Feature prioritization matrix
- ✅ User stories with acceptance criteria
- ✅ Success metrics definition
- ✅ Release plan (3-week timeline)

---

### 2. **UX Researcher** - User Needs & Interface Design

**Responsibilities:**
- Research user workflows
- Identify pain points in current CLI usage
- Design user-friendly abstractions for complex parameters
- Create usability testing plan

**Tasks:**

#### Research Phase
```yaml
user_research:
  interview_questions:
    - "What CLI commands do you use most frequently?"
    - "Which parameters are confusing?"
    - "What's missing from the current GUI?"
    - "Do you prefer presets or manual configuration?"

  usability_testing:
    tasks:
      - "Index a document with custom chunk size"
      - "Enable HyDE retrieval and test query"
      - "Configure semantic cache settings"
      - "View performance metrics"

  findings:
    pain_points:
      - Too many environment variables (30+)
      - No visibility into what's enabled/disabled
      - Confusion about hybrid search values
      - Missing performance feedback
      - No way to compare configurations

    user_preferences:
      - Presets for common scenarios
      - Visual feedback for complex settings
      - One-click toggle for advanced features
      - Performance comparison tools
      - Configuration export/import
```

#### UX Recommendations
```yaml
design_patterns:
  progressive_disclosure:
    level_1: "Presets only (Fast/Balanced/Quality)"
    level_2: "Basic parameters (chunk size, model)"
    level_3: "Advanced features (expandable sections)"
    level_4: "Expert mode (all raw parameters)"

  visual_hierarchy:
    critical_settings: "Always visible with validation"
    common_settings: "Visible in default view"
    advanced_settings: "Collapsed expanders"
    expert_settings: "Separate tab or toggle"

  feedback_systems:
    real_time_validation: "Show warnings for invalid configs"
    impact_preview: "Estimate effect of parameter changes"
    performance_tracking: "Show before/after metrics"
    configuration_comparison: "Compare preset vs custom"
```

**Deliverables:**
- ✅ User research findings document
- ✅ Usability testing report
- ✅ UX recommendations document
- ✅ Wireframes for new features

---

### 3. **UI Designer** - Interface Design & Visual Hierarchy

**Responsibilities:**
- Design layouts for new features
- Create visual components
- Ensure consistent design language
- Optimize for usability

**Tasks:**

#### Design System Extension
```yaml
new_components:
  mode_selector:
    description: "Index-only / Query-only / Full pipeline selector"
    design: "Radio buttons or segmented control"
    location: "Top of Quick Start and Advanced Index pages"

  performance_dashboard:
    description: "Real-time and historical performance metrics"
    design: "Card-based layout with charts"
    components:
      - Timing breakdown chart (already implemented ✅)
      - Throughput metrics
      - Historical trends
      - Configuration comparison

  advanced_features_panel:
    description: "Organized advanced parameters"
    design: "Tabbed interface or accordion"
    sections:
      - Query Enhancement (Expansion, HyDE)
      - Result Processing (Reranking, MMR)
      - Caching (Semantic cache settings)
      - Conversation (Memory, summarization)

  configuration_presets:
    description: "Scenario-based configuration templates"
    design: "Card selection with descriptions"
    presets:
      - Speed Optimized (cache ON, expansion OFF)
      - Quality Optimized (reranking ON, HyDE ON)
      - Balanced (current defaults)
      - Chat Optimized (conversation memory ON)

  parameter_validator:
    description: "Real-time parameter validation and suggestions"
    design: "Inline warnings and success indicators"
    features:
      - Red warning for problematic values
      - Green checkmark for recommended values
      - Yellow info for sub-optimal but valid
```

#### Layout Mockups
```
┌─────────────────────────────────────────────────────────┐
│ 🚀 Quick Start                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ 📋 Mode Selection                                       │
│ ○ Full Pipeline (Index + Query)                         │
│ ○ Index Only                                            │
│ ○ Query Only                                            │
│                                                          │
│ 1. Choose Preset: [Fast ⚡] [Balanced ⚖️] [Quality 🎯]  │
│                                                          │
│ 2. Select Document: [dropdown]                          │
│                                                          │
│ 3. Advanced Options (Optional) ▼                        │
│    ├─ Chunking                                          │
│    ├─ Embedding                                         │
│    ├─ Retrieval Enhancement                            │
│    │  ├─ ☑ Enable Query Expansion (method: [llm ▼])   │
│    │  ├─ ☑ Enable HyDE (hypotheses: [1 ▼])            │
│    │  ├─ ☑ Enable Reranking (candidates: [12])        │
│    │  └─ MMR Diversity: [0.0 ───●─── 1.0]             │
│    ├─ Caching                                           │
│    │  ├─ ☑ Semantic Cache (threshold: [0.92])         │
│    │  └─ Max size: [1000] TTL: [86400s]               │
│    └─ Conversation                                      │
│       ├─ ☑ Enable Conversation Memory                  │
│       └─ Max turns: [10] Timeout: [3600s]             │
│                                                          │
│ 4. [🚀 Start Indexing]                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 📊 Performance Dashboard (NEW TAB)                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│ Current Session:                                         │
│ ┌──────────────┬──────────────┬──────────────┐         │
│ │ Indexed      │ Queried      │ Cache Hits   │         │
│ │ 19,820 chunks│ 15 queries   │ 3/15 (20%)   │         │
│ └──────────────┴──────────────┴──────────────┘         │
│                                                          │
│ Timing Breakdown:                                        │
│ [Bar Chart showing Load/Chunk/Embed/Store]              │
│                                                          │
│ Query Performance:                                       │
│ [Line chart showing query latency over time]            │
│                                                          │
│ Cache Statistics:                                        │
│ [Pie chart: Hits vs Misses vs Bypassed]                │
└─────────────────────────────────────────────────────────┘
```

**Deliverables:**
- ✅ Design mockups for all new features
- ✅ Component library extensions
- ✅ Style guide updates
- ✅ Accessibility review

---

### 4. **Frontend Developer** - UI Implementation

**Responsibilities:**
- Implement new UI components
- Add missing parameter controls
- Create new tabs/pages
- Integrate with backend

**Tasks:**

#### Phase 1: Parameter Controls (3-5 days)
```python
# File: rag_web_enhanced.py

# Task 1.1: Add Mode Selector
def add_mode_selector():
    """Add index-only/query-only/full pipeline mode selector."""
    mode = st.radio(
        "Pipeline Mode:",
        ["Full Pipeline", "Index Only", "Query Only"],
        help="Choose what operations to perform"
    )
    return mode

# Task 1.2: Add MMR Diversity Control
def add_mmr_control():
    """Add MMR diversity slider to Advanced Features."""
    mmr = st.slider(
        "MMR Diversity",
        0.0, 1.0, 0.0, 0.1,
        help="0.0=most relevant, 1.0=most diverse"
    )
    if mmr > 0:
        st.info(f"✅ Diversity enabled: {mmr:.1f}")
    return mmr

# Task 1.3: Add Query Expansion Configuration
def add_query_expansion_config():
    """Add query expansion method and count controls."""
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Expansion Method",
            ["llm", "keyword", "multi"],
            help="LLM=best quality, keyword=fastest"
        )
    with col2:
        count = st.slider("Expansion Count", 1, 5, 2)
    return method, count

# Task 1.4: Add Reranking Configuration
def add_reranking_config():
    """Add reranking parameter controls."""
    col1, col2, col3 = st.columns(3)
    with col1:
        candidates = st.number_input("Rerank Candidates", 5, 50, 12)
    with col2:
        top_k = st.number_input("Rerank TOP_K", 1, 20, 4)
    with col3:
        model = st.selectbox(
            "Rerank Model",
            [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
            ]
        )
    return candidates, top_k, model

# Task 1.5: Add Semantic Cache Controls
def add_cache_controls():
    """Add semantic cache configuration."""
    with st.expander("🗄️ Semantic Cache", expanded=False):
        enable = st.checkbox("Enable Semantic Cache", value=True)

        if enable:
            col1, col2, col3 = st.columns(3)
            with col1:
                threshold = st.slider("Similarity Threshold", 0.80, 0.99, 0.92, 0.01)
            with col2:
                max_size = st.number_input("Max Cached Queries", 100, 10000, 1000, 100)
            with col3:
                ttl = st.number_input("TTL (seconds)", 0, 864000, 86400, 3600)

            # Show cache stats if available
            if 'cache_hits' in st.session_state:
                hits = st.session_state.cache_hits
                total = st.session_state.total_queries
                hit_rate = 100 * hits / total if total > 0 else 0
                st.metric("Cache Hit Rate", f"{hit_rate:.1f}%", f"{hits}/{total}")

        return enable, threshold if enable else 0, max_size if enable else 0, ttl if enable else 0

# Task 1.6: Add HyDE Controls
def add_hyde_controls():
    """Add HyDE retrieval configuration."""
    with st.expander("🔬 HyDE (Hypothetical Document Embeddings)", expanded=False):
        st.caption("Advanced: Generates hypothetical answers before retrieval")

        enable = st.checkbox("Enable HyDE", value=False)

        if enable:
            col1, col2, col3 = st.columns(3)
            with col1:
                num_hyp = st.slider("Hypotheses", 1, 3, 1)
            with col2:
                length = st.slider("Hypothesis Length", 50, 200, 100, 10)
            with col3:
                fusion = st.selectbox("Fusion Method", ["rrf", "avg", "max"])

            st.info(f"⚠️ +{num_hyp * 100}ms latency, +10-20% quality")

        return enable, num_hyp if enable else 1, length if enable else 100, fusion if enable else 'rrf'

# Task 1.7: Add Conversation Memory
def add_conversation_memory():
    """Add conversation memory controls."""
    with st.expander("💬 Conversation Memory", expanded=False):
        st.caption("Multi-turn dialogue with context awareness")

        enable = st.checkbox("Enable Conversation Mode", value=False)

        if enable:
            col1, col2 = st.columns(2)
            with col1:
                max_turns = st.slider("Max Turns", 3, 20, 10)
                auto_summarize = st.checkbox("Auto Summarize", value=True)
            with col2:
                timeout = st.number_input("Timeout (seconds)", 300, 7200, 3600, 300)
                summarize_threshold = st.slider("Summarize After", 3, 10, 5) if auto_summarize else 5

            # Show conversation history
            if 'conversation_history' in st.session_state:
                st.caption(f"Current conversation: {len(st.session_state.conversation_history)} turns")

        return enable, max_turns if enable else 10, timeout if enable else 3600, \
               auto_summarize if enable else False, summarize_threshold if enable else 5
```

#### Phase 2: New Pages/Tabs (5-7 days)
```python
# Task 2.1: Create Performance Dashboard Tab
def page_performance():
    """Performance monitoring and analytics page."""
    st.header("📊 Performance Dashboard")

    # Session stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Indexed", "35,188 chunks")
    with col2:
        st.metric("Queries Run", "47")
    with col3:
        st.metric("Cache Hit Rate", "23.4%")
    with col4:
        st.metric("Avg Query Time", "1.2s")

    # Timing breakdown (reuse existing code)
    st.subheader("Indexing Performance")
    # Show last indexing timing breakdown

    # Query performance over time
    st.subheader("Query Performance Trends")
    # Line chart of query latency

    # Configuration impact analysis
    st.subheader("Configuration Impact")
    # Compare different configurations

# Task 2.2: Create Configuration Presets Tab
def page_presets():
    """Configuration presets and templates page."""
    st.header("⚙️ Configuration Presets")

    presets = {
        "Speed Optimized": {
            "description": "Fastest responses, good quality",
            "config": {
                "HYBRID_ALPHA": 1.0,
                "ENABLE_QUERY_EXPANSION": False,
                "ENABLE_RERANKING": False,
                "ENABLE_SEMANTIC_CACHE": True,
                "CACHE_THRESHOLD": 0.85,
            }
        },
        "Quality Optimized": {
            "description": "Best quality, slower",
            "config": {
                "ENABLE_QUERY_EXPANSION": True,
                "ENABLE_RERANKING": True,
                "ENABLE_HYDE": True,
                "MMR_THRESHOLD": 0.5,
            }
        },
        # ... more presets
    }

    # Allow users to save/load configurations

# Task 2.3: Add Chat-Style Query Interface
def page_chat():
    """Chat-style interface with conversation memory."""
    st.header("💬 Chat Mode")

    # Show conversation history
    for msg in st.session_state.get('conversation', []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add to conversation
        # Run query with conversation context
        # Display with streaming if available
        pass
```

#### Phase 3: Integration & Polish (3-4 days)
- Integrate all new components
- Add keyboard shortcuts
- Implement configuration export/import
- Add help tooltips throughout

**Deliverables:**
- ✅ New UI components implemented
- ✅ New tabs/pages added
- ✅ Integration complete
- ✅ UI testing complete

---

### 5. **Python Pro** - Backend Implementation

**Responsibilities:**
- Implement missing backend functionality
- Add API endpoints for new features
- Optimize performance
- Ensure thread safety

**Tasks:**

#### Backend Feature Implementation
```python
# File: rag_web_backend.py (NEW FILE)

import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class PerformanceMetrics:
    """Track performance metrics for dashboard."""
    indexing_times: List[float]
    query_times: List[float]
    cache_hits: int
    cache_misses: int
    total_chunks_indexed: int
    total_queries: int

    def get_cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return 100 * self.cache_hits / total if total > 0 else 0

class ConfigurationManager:
    """Manage configuration presets and user settings."""

    def __init__(self):
        self.presets = self._load_presets()
        self.custom_configs = {}

    def save_configuration(self, name: str, config: Dict):
        """Save user configuration for later reuse."""
        self.custom_configs[name] = config
        # Persist to file
        with open('.cache/configs.json', 'w') as f:
            json.dump(self.custom_configs, f, indent=2)

    def load_configuration(self, name: str) -> Dict:
        """Load saved configuration."""
        return self.custom_configs.get(name)

    def export_to_env(self, config: Dict) -> str:
        """Export configuration as .env format."""
        lines = []
        for key, value in config.items():
            lines.append(f"{key}={value}")
        return "\\n".join(lines)

class ConversationManager:
    """Manage multi-turn conversations with memory."""

    def __init__(self, max_turns: int = 10):
        self.conversations: Dict[str, List[Dict]] = {}
        self.max_turns = max_turns

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a conversation turn."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

        # Trim to max turns
        if len(self.conversations[session_id]) > self.max_turns:
            self.conversations[session_id] = self.conversations[session_id][-self.max_turns:]

    def get_context(self, session_id: str, last_n: int = 5) -> str:
        """Get conversation context for query reformulation."""
        if session_id not in self.conversations:
            return ""

        recent = self.conversations[session_id][-last_n:]
        context = "\\n".join([f"{t['role']}: {t['content']}" for t in recent])
        return context

class CacheManager:
    """Manage semantic cache with statistics."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.cache_entries = []

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = 100 * self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "entries": len(self.cache_entries)
        }

# Initialize global managers
performance_metrics = PerformanceMetrics(
    indexing_times=[],
    query_times=[],
    cache_hits=0,
    cache_misses=0,
    total_chunks_indexed=0,
    total_queries=0
)

config_manager = ConfigurationManager()
conversation_manager = ConversationManager()
cache_manager = CacheManager()
```

#### Integration with Existing Code
```python
# Task: Modify run_indexing() to support index-only mode
def run_indexing(doc_path, table_name, ..., mode="full"):
    """
    mode: "full" | "index-only" | "query-only"
    """
    if mode in ["full", "index-only"]:
        # Do indexing
        pass

    if mode in ["full", "query-only"]:
        # Do querying (if not index-only)
        pass

# Task: Modify run_query() to use conversation context
def run_query(table_name, query, ..., use_conversation=False, session_id=None):
    if use_conversation and session_id:
        # Get conversation context
        context = conversation_manager.get_context(session_id)

        # Reformulate query with context
        reformulated = reformulate_with_context(query, context)

        # Use reformulated query
        query = reformulated

    # Continue with normal query...
```

**Deliverables:**
- ✅ Backend managers implemented
- ✅ Integration with existing code
- ✅ Performance tracking added
- ✅ Thread-safe operations

---

### 6. **Refactoring Specialist** - Code Quality & Architecture

**Responsibilities:**
- Refactor duplicated code
- Improve maintainability
- Extract reusable components
- Optimize performance

**Tasks:**

#### Refactoring Plan
```python
# Current issues:
# 1. Parameter handling scattered across multiple functions
# 2. Duplicated validation logic
# 3. Hard-coded parameter lists
# 4. No centralized configuration management

# Refactoring 1: Centralized Parameter Registry
class ParameterRegistry:
    """Central registry of all configurable parameters."""

    @dataclass
    class Parameter:
        name: str
        type: type
        default: any
        min_val: Optional[any] = None
        max_val: Optional[any] = None
        help_text: str = ""
        category: str = "general"
        ui_component: str = "slider"  # slider, checkbox, selectbox, number_input
        options: Optional[List] = None  # For selectbox

    PARAMETERS = {
        # Chunking
        "CHUNK_SIZE": Parameter(
            name="Chunk Size",
            type=int,
            default=700,
            min_val=100,
            max_val=3000,
            help_text="Characters per chunk",
            category="chunking",
            ui_component="slider"
        ),
        "CHUNK_OVERLAP": Parameter(
            name="Chunk Overlap",
            type=int,
            default=150,
            min_val=0,
            max_val=500,
            help_text="Overlap between chunks",
            category="chunking",
            ui_component="slider"
        ),

        # Retrieval Enhancement
        "MMR_THRESHOLD": Parameter(
            name="MMR Diversity",
            type=float,
            default=0.0,
            min_val=0.0,
            max_val=1.0,
            help_text="0=most relevant, 1=most diverse",
            category="retrieval",
            ui_component="slider"
        ),

        # Query Expansion
        "QUERY_EXPANSION_METHOD": Parameter(
            name="Expansion Method",
            type=str,
            default="llm",
            options=["llm", "keyword", "multi"],
            help_text="LLM=best, keyword=fast",
            category="query_expansion",
            ui_component="selectbox"
        ),

        # ... all 50+ parameters defined here
    }

    @classmethod
    def render_ui_control(cls, param_key: str, current_value: any = None):
        """Render appropriate UI control for parameter."""
        param = cls.PARAMETERS[param_key]
        value = current_value or param.default

        if param.ui_component == "slider":
            return st.slider(
                param.name,
                param.min_val,
                param.max_val,
                value,
                help=param.help_text
            )
        elif param.ui_component == "checkbox":
            return st.checkbox(param.name, value=value, help=param.help_text)
        elif param.ui_component == "selectbox":
            return st.selectbox(param.name, param.options, index=param.options.index(value), help=param.help_text)
        elif param.ui_component == "number_input":
            return st.number_input(param.name, param.min_val, param.max_val, value, help=param.help_text)

    @classmethod
    def get_by_category(cls, category: str) -> Dict:
        """Get all parameters in a category."""
        return {k: v for k, v in cls.PARAMETERS.items() if v.category == category}

# Refactoring 2: Configuration Validator
class ConfigurationValidator:
    """Validate parameter combinations and suggest fixes."""

    @staticmethod
    def validate(config: Dict) -> List[str]:
        """Return list of warnings/errors."""
        warnings = []

        # Check chunk overlap ratio
        if config.get('CHUNK_OVERLAP', 0) >= config.get('CHUNK_SIZE', 700):
            warnings.append("⚠️ CHUNK_OVERLAP must be less than CHUNK_SIZE")

        # Check embedding dimension matches model
        model = config.get('EMBED_MODEL', '')
        dim = config.get('EMBED_DIM', 384)
        expected = cls._get_expected_dim(model)
        if dim != expected:
            warnings.append(f"⚠️ EMBED_DIM={dim} but {model} uses {expected} dimensions")

        # Check problematic combinations
        if config.get('HYBRID_ALPHA', 1.0) < 1.0 and not cls._check_bm25_installed():
            warnings.append("⚠️ HYBRID_ALPHA < 1.0 requires rank-bm25: pip install rank-bm25")

        return warnings

# Refactoring 3: Extract Common UI Patterns
class UIComponents:
    """Reusable UI component library."""

    @staticmethod
    def parameter_group(title: str, parameters: List[str], expanded: bool = True):
        """Render a group of related parameters."""
        with st.expander(title, expanded=expanded):
            values = {}
            for param_key in parameters:
                values[param_key] = ParameterRegistry.render_ui_control(param_key)
            return values

    @staticmethod
    def validation_display(warnings: List[str]):
        """Display validation warnings."""
        if warnings:
            for warning in warnings:
                if "⚠️" in warning:
                    st.warning(warning)
                elif "❌" in warning:
                    st.error(warning)
                elif "✅" in warning:
                    st.success(warning)

    @staticmethod
    def metric_card(title: str, value: str, delta: Optional[str] = None):
        """Styled metric card."""
        st.metric(title, value, delta=delta)

# Usage in pages:
def page_advanced_index_refactored():
    """Refactored advanced indexing page."""
    st.header("⚙️ Advanced Indexing")

    # Chunking parameters (auto-rendered from registry)
    chunking_params = UIComponents.parameter_group(
        "Chunking Parameters",
        ["CHUNK_SIZE", "CHUNK_OVERLAP"]
    )

    # Embedding parameters
    embedding_params = UIComponents.parameter_group(
        "Embedding Configuration",
        ["EMBED_MODEL", "EMBED_DIM", "EMBED_BATCH", "EMBED_BACKEND"]
    )

    # Validate configuration
    config = {**chunking_params, **embedding_params}
    warnings = ConfigurationValidator.validate(config)
    UIComponents.validation_display(warnings)
```

**Deliverables:**
- ✅ Parameter registry system
- ✅ Configuration validator
- ✅ Reusable UI components
- ✅ Refactored pages using new patterns

---

### 7. **Agent Organizer** - Project Coordination

**Responsibilities:**
- Coordinate between agents
- Manage dependencies
- Track progress
- Ensure quality

**Tasks:**

#### Project Organization
```yaml
sprint_1_week_1:
  objectives:
    - Implement missing parameter controls
    - Add mode selector (index-only/query-only)
    - Create performance dashboard foundation

  agent_coordination:
    day_1_2:
      product_manager: "Define requirements for missing features"
      ux_researcher: "User research and workflow analysis"

    day_3_4:
      ui_designer: "Design mockups for new controls"
      python_pro: "Implement backend managers"

    day_5:
      frontend_developer: "Implement parameter controls"
      refactoring_specialist: "Begin parameter registry refactor"

  deliverables:
    - Parameter registry system
    - Mode selector UI
    - Backend managers (Config, Cache, Conversation)

sprint_2_week_2:
  objectives:
    - Complete advanced feature controls
    - Implement performance dashboard
    - Add configuration presets

  agent_coordination:
    day_1_2:
      frontend_developer: "Implement HyDE, MMR, advanced controls"
      python_pro: "Performance tracking backend"

    day_3_4:
      ui_designer: "Performance dashboard design"
      frontend_developer: "Dashboard implementation"

    day_5:
      refactoring_specialist: "Refactor existing pages"
      all_agents: "Integration testing"

  deliverables:
    - All advanced controls implemented
    - Performance dashboard live
    - Configuration presets

sprint_3_week_3:
  objectives:
    - Conversation memory/chat mode
    - Configuration export/import
    - Polish and testing

  agent_coordination:
    day_1_2:
      python_pro: "Conversation manager backend"
      ui_designer: "Chat interface design"

    day_3_4:
      frontend_developer: "Chat mode implementation"
      refactoring_specialist: "Code cleanup and optimization"

    day_5:
      all_agents: "Testing, bug fixes, documentation"

  deliverables:
    - Chat mode complete
    - Export/import functionality
    - Full testing and documentation
```

**Progress Tracking:**
- Daily standup (async)
- Weekly demos
- Issue tracking via GitHub/Notion
- Quality gates before merges

**Deliverables:**
- ✅ Sprint plans
- ✅ Coordination schedule
- ✅ Progress dashboards
- ✅ Quality assurance reports

---

## 📋 Complete Feature Matrix

### Priority 1: High Impact, Low Complexity (Week 1)

| Feature | CLI | GUI Location | Implementation Effort | Agent |
|---------|-----|--------------|----------------------|-------|
| Index-only mode | `--index-only` | Quick Start (radio button) | 2 hours | Frontend |
| Query-only mode | `--query-only` | Query page (toggle) | 2 hours | Frontend |
| Semantic cache enable | `ENABLE_SEMANTIC_CACHE=1` | Query page | 1 hour | Frontend |
| Cache threshold | `SEMANTIC_CACHE_THRESHOLD=0.92` | Query page | 1 hour | Frontend |
| Cache max size | `SEMANTIC_CACHE_MAX_SIZE=1000` | Query page | 30 min | Frontend |
| Cache TTL | `SEMANTIC_CACHE_TTL=86400` | Query page | 30 min | Frontend |
| MMR diversity | `MMR_THRESHOLD=0.5` | Query page | 1 hour | Frontend |
| Chat metadata extract | `EXTRACT_CHAT_METADATA=1` | Indexing pages | 1 hour | Frontend |
| Performance metrics display | Built-in logging | New tab | 4 hours | Frontend + Designer |

**Total: ~15 hours** (2 days)

### Priority 2: Medium Impact, Medium Complexity (Week 2)

| Feature | CLI | GUI Location | Implementation Effort | Agent |
|---------|-----|--------------|----------------------|-------|
| Query expansion method | `QUERY_EXPANSION_METHOD=llm` | Query page | 2 hours | Frontend |
| Query expansion count | `QUERY_EXPANSION_COUNT=2` | Query page | 1 hour | Frontend |
| Rerank candidates | `RERANK_CANDIDATES=12` | Query page | 1 hour | Frontend |
| Rerank TOP_K | `RERANK_TOP_K=4` | Query page | 1 hour | Frontend |
| Rerank model | `RERANK_MODEL=model` | Query page | 2 hours | Frontend |
| HyDE enable | `ENABLE_HYDE=1` | Query page | 3 hours | Frontend + Python |
| HyDE num hypotheses | `HYDE_NUM_HYPOTHESES=2` | Query page | 1 hour | Frontend |
| HyDE hypothesis length | `HYDE_HYPOTHESIS_LENGTH=100` | Query page | 1 hour | Frontend |
| HyDE fusion method | `HYDE_FUSION_METHOD=rrf` | Query page | 1 hour | Frontend |
| LLM model URL | `MODEL_URL=url` | Settings page | 3 hours | Frontend |
| LLM model path | `MODEL_PATH=path` | Settings page | 2 hours | Frontend |
| GPU layers | `N_GPU_LAYERS=24` | Settings page | 1 hour | Frontend |
| Batch size | `N_BATCH=256` | Settings page | 1 hour | Frontend |
| Configuration export | N/A (new feature) | Settings page | 4 hours | Frontend + Python |
| Configuration import | N/A (new feature) | Settings page | 4 hours | Frontend + Python |

**Total: ~30 hours** (4 days)

### Priority 3: Advanced Features, High Complexity (Week 3)

| Feature | CLI | GUI Location | Implementation Effort | Agent |
|---------|-----|--------------|----------------------|-------|
| Interactive REPL | `--interactive` | New Chat tab | 8 hours | Frontend + Python |
| Conversation memory | `ENABLE_CONVERSATION_MEMORY=1` | Chat tab | 6 hours | Python + Frontend |
| Max conversation turns | `MAX_CONVERSATION_TURNS=10` | Chat tab | 1 hour | Frontend |
| Conversation timeout | `CONVERSATION_TIMEOUT=3600` | Chat tab | 2 hours | Python |
| Auto summarize | `AUTO_SUMMARIZE=1` | Chat tab | 4 hours | Python |
| Summarize threshold | `SUMMARIZE_THRESHOLD=5` | Chat tab | 1 hour | Frontend |
| vLLM backend support | vLLM scripts | Settings page | 12 hours | Python Pro |
| Batch multi-document indexing | N/A (new) | Advanced Index | 6 hours | Frontend + Python |
| Query result export | N/A (new) | Query page | 3 hours | Frontend |
| Index comparison tool | N/A (new) | Performance tab | 8 hours | Frontend + Designer |

**Total: ~51 hours** (6-7 days)

---

## 🎨 Detailed Implementation Specifications

### Feature 1: Index-Only / Query-Only Mode

**UI Designer:**
```python
# Location: Quick Start page, Advanced Index page
mode_options = ["Full Pipeline", "Index Only", "Query Only"]
mode = st.radio("Pipeline Mode:", mode_options, horizontal=True)

if mode == "Index Only":
    st.info("✅ Will index documents but skip querying")
elif mode == "Query Only":
    st.info("✅ Will use existing index without re-indexing")
```

**Frontend Developer:**
```python
# Implementation
def run_pipeline_with_mode(mode, ...):
    if mode in ["Full Pipeline", "Index Only"]:
        run_indexing(...)

    if mode in ["Full Pipeline", "Query Only"]:
        run_query(...)
```

**Testing:**
- ✅ Index-only completes without query
- ✅ Query-only skips indexing
- ✅ Full pipeline runs both

---

### Feature 2: Semantic Cache Configuration

**UI Designer:**
```python
# Location: Query page → Advanced Features
with st.expander("🗄️ Semantic Cache"):
    enable = st.checkbox("Enable Cache", value=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        threshold = st.slider("Threshold", 0.80, 0.99, 0.92, 0.01)
        st.caption(f"Hit rate: ~{estimate_hit_rate(threshold):.0f}%")

    with col2:
        max_size = st.number_input("Max Size", 100, 10000, 1000)

    with col3:
        ttl_hours = st.number_input("TTL (hours)", 0, 72, 24)
        ttl = ttl_hours * 3600

    # Live stats
    if st.session_state.get('cache_stats'):
        stats = st.session_state.cache_stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Hits", stats['hits'])
        col2.metric("Misses", stats['misses'])
        col3.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
```

**Python Pro:**
```python
# Backend integration
def run_query_with_cache(query, cache_config):
    if cache_config['enabled']:
        # Check cache
        cached = check_semantic_cache(query, cache_config['threshold'])
        if cached:
            cache_manager.record_hit()
            return cached

        cache_manager.record_miss()

    # Execute query
    result = execute_query(query)

    # Cache result
    if cache_config['enabled']:
        add_to_cache(query, result, cache_config['ttl'])

    return result
```

---

### Feature 3: MMR Diversity Control

**UI Designer:**
```python
# Location: Query page → Advanced Features
mmr = st.slider(
    "MMR Diversity",
    0.0, 1.0, 0.0, 0.1,
    help="0.0=most relevant, 1.0=most diverse, prevents repetitive chunks"
)

if mmr > 0:
    st.success(f"✅ Diversity enabled: {mmr:.1f} - will avoid similar chunks")
else:
    st.info("⏹️ Disabled - returns most relevant chunks (may be similar)")

# Visual example
if mmr > 0.5:
    st.caption("High diversity: Good for broad topic exploration")
elif mmr > 0:
    st.caption("Moderate diversity: Balanced relevance and variety")
```

**Frontend Developer:**
```python
# Pass to query function
def run_query(..., mmr_threshold):
    rag.S.mmr_threshold = mmr_threshold
    # Continue with query...
```

---

### Feature 4: Query Expansion Configuration

**UI Designer:**
```python
# Location: Query page → Advanced Features → Query Expansion
enable_expansion = st.checkbox("Enable Query Expansion")

if enable_expansion:
    col1, col2 = st.columns(2)

    with col1:
        method = st.selectbox(
            "Method",
            ["llm", "keyword", "multi"],
            help="llm=best quality, keyword=fastest"
        )

        if method == "llm":
            st.warning("⚠️ Downloads 4.4GB model on first use")
        elif method == "keyword":
            st.success("✅ Fast, no model download")

    with col2:
        count = st.slider("Expansions", 1, 5, 2)

    # Show example
    with st.expander("💡 Example"):
        st.text("Original: 'Londres'")
        st.text("Expansion 1: 'What is the capital of UK?'")
        st.text("Expansion 2: 'Information about London'")

    est_time = "1-3s" if method in ["llm", "multi"] else "<0.1s"
    st.caption(f"⏱️ Added latency: {est_time}")
```

---

### Feature 5: HyDE Retrieval

**UI Designer:**
```python
# Location: Query page → Advanced Features
with st.expander("🔬 HyDE (Hypothetical Document Embeddings)"):
    st.caption("⚡ Advanced: Generates hypothetical answers before retrieval")

    enable = st.checkbox("Enable HyDE")

    if enable:
        col1, col2, col3 = st.columns(3)

        with col1:
            num = st.slider("Hypotheses", 1, 3, 1)
            st.caption(f"+{num * 100}ms latency")

        with col2:
            length = st.slider("Length (tokens)", 50, 200, 100, 10)

        with col3:
            fusion = st.selectbox("Fusion", ["rrf", "avg", "max"])

        # Quality impact visualization
        st.progress(0.15, text="Quality improvement: +10-20%")

        # Show how it works
        with st.expander("How HyDE Works"):
            st.markdown("""
            1. Your query: "What is attention mechanism?"
            2. LLM generates answer: "Attention mechanism is a neural network component..."
            3. Embeds the generated answer (not your question)
            4. Retrieves documents similar to the answer
            5. Better matching since documents are answer-style, not question-style
            """)
```

---

### Feature 6: Conversation Memory / Chat Mode

**UI Designer:**
```python
# New tab: Chat Mode
def page_chat():
    """Chat-style interface with conversation memory."""
    st.header("💬 Chat Mode")

    # Configuration sidebar
    with st.sidebar:
        st.subheader("Conversation Settings")

        enable_memory = st.checkbox("Enable Memory", value=True)
        max_turns = st.slider("Max Turns", 3, 20, 10)
        auto_summarize = st.checkbox("Auto Summarize", value=True)

        if st.button("🗑️ Clear History"):
            st.session_state.conversation = []
            st.rerun()

        # Stats
        if 'conversation' in st.session_state:
            st.metric("Current Turns", len(st.session_state.conversation))

    # Chat interface
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Display conversation history
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.caption(f"- {src}")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.conversation.append({"role": "user", "content": prompt})

        # Get conversation context
        context = get_conversation_context(st.session_state.conversation)

        # Reformulate query with context
        reformulated = reformulate_query(prompt, context)

        # Run query
        with st.spinner("Thinking..."):
            result = run_query_with_memory(reformulated, context)

        # Add assistant message
        st.session_state.conversation.append({
            "role": "assistant",
            "content": result['answer'],
            "sources": result['sources']
        })

        st.rerun()
```

**Python Pro:**
```python
# Backend: Conversation memory implementation
class ConversationMemory:
    def __init__(self, max_turns=10, auto_summarize=True, summarize_threshold=5):
        self.max_turns = max_turns
        self.auto_summarize = auto_summarize
        self.summarize_threshold = summarize_threshold
        self.conversations = {}

    def reformulate_query(self, query: str, conversation_history: List[Dict]) -> str:
        """Reformulate query with conversation context."""
        if not conversation_history:
            return query

        # Extract entities and topics from history
        entities = self._extract_entities(conversation_history)
        last_topic = conversation_history[-1]['content'] if conversation_history else ""

        # Use LLM to reformulate
        prompt = f"""
        Given this conversation history:
        {self._format_history(conversation_history[-3:])}

        Reformulate this query with context:
        "{query}"

        Output only the reformulated query.
        """

        reformulated = llm.complete(prompt).text.strip()
        return reformulated

    def should_summarize(self, conversation_history: List[Dict]) -> bool:
        """Check if conversation should be summarized."""
        return (self.auto_summarize and
                len(conversation_history) >= self.summarize_threshold)

    def summarize_conversation(self, conversation_history: List[Dict]) -> str:
        """Summarize old conversation turns."""
        # Use LLM to create concise summary
        history_text = self._format_history(conversation_history)

        prompt = f"""
        Summarize this conversation in 2-3 sentences:
        {history_text}

        Focus on key topics and entities discussed.
        """

        summary = llm.complete(prompt).text.strip()
        return summary
```

---

### Feature 7: Performance Dashboard

**UI Designer:**
```python
def page_performance():
    """Comprehensive performance dashboard."""
    st.header("📊 Performance Dashboard")

    # Load metrics
    metrics = load_performance_metrics()

    # Overview cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sessions", metrics['total_sessions'])
    with col2:
        st.metric("Total Queries", metrics['total_queries'])
    with col3:
        st.metric("Avg Query Time", f"{metrics['avg_query_time']:.2f}s")
    with col4:
        st.metric("Cache Hit Rate", f"{metrics['cache_hit_rate']:.1f}%")

    # Time series charts
    st.subheader("Query Performance Over Time")
    fig = px.line(
        metrics['query_history'],
        x='timestamp',
        y='latency',
        title='Query Latency (Last 24 hours)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Configuration impact analysis
    st.subheader("Configuration Impact")

    config_comparison = pd.DataFrame({
        'Configuration': ['Pure Vector', 'Hybrid α=0.7', 'With Expansion', 'With Reranking'],
        'Avg Latency (s)': [0.8, 2.3, 8.5, 1.5],
        'Quality Score': [0.75, 0.78, 0.82, 0.85],
        'Cache Hit Rate (%)': [25, 18, 10, 22]
    })

    fig = px.scatter(
        config_comparison,
        x='Avg Latency (s)',
        y='Quality Score',
        size='Cache Hit Rate (%)',
        text='Configuration',
        title='Configuration Trade-offs'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.subheader("Detailed Metrics")
    st.dataframe(metrics['detailed_stats'], use_container_width=True)
```

---

### Feature 8: Configuration Management

**Frontend Developer:**
```python
def page_configuration_manager():
    """Configuration save/load/export functionality."""
    st.header("⚙️ Configuration Manager")

    # Current configuration display
    st.subheader("Current Configuration")
    current_config = get_current_config()

    # Show as JSON
    with st.expander("View as JSON"):
        st.json(current_config)

    # Save configuration
    st.subheader("Save Configuration")
    config_name = st.text_input("Configuration Name")
    if st.button("💾 Save"):
        save_configuration(config_name, current_config)
        st.success(f"Saved '{config_name}'")

    # Load configuration
    st.subheader("Load Configuration")
    saved_configs = list_saved_configurations()
    if saved_configs:
        selected = st.selectbox("Saved Configurations", saved_configs)
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Load"):
                load_configuration(selected)
                st.success(f"Loaded '{selected}'")
                st.rerun()

        with col2:
            if st.button("🗑️ Delete"):
                delete_configuration(selected)
                st.success(f"Deleted '{selected}'")
                st.rerun()

    # Export/Import
    st.subheader("Export/Import")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Export as .env file")
        env_content = export_to_env(current_config)
        st.download_button(
            "📥 Download .env",
            env_content,
            file_name="rag_config.env",
            mime="text/plain"
        )

    with col2:
        st.caption("Import from .env file")
        uploaded = st.file_uploader("Upload .env", type=['env', 'txt'])
        if uploaded:
            content = uploaded.read().decode()
            parsed_config = parse_env_file(content)
            if st.button("Apply Configuration"):
                apply_configuration(parsed_config)
                st.success("Configuration applied!")
                st.rerun()
```

---

## 🏗️ Architecture Changes

### New File Structure
```
rag_web_enhanced.py          # Main UI (existing)
rag_web_backend.py           # New: Backend managers
rag_web_components.py        # New: Reusable UI components
rag_web_config.py            # New: Configuration management
rag_web_performance.py       # New: Performance tracking
rag_web_conversation.py      # New: Conversation memory
```

### Refactored Code Organization
```python
# rag_web_components.py
class ParameterControls:
    @staticmethod
    def render_chunking_controls():
        """Reusable chunking parameter controls."""
        pass

    @staticmethod
    def render_embedding_controls():
        """Reusable embedding parameter controls."""
        pass

    @staticmethod
    def render_retrieval_controls():
        """Reusable retrieval parameter controls."""
        pass

# rag_web_config.py
class ConfigurationManager:
    def save(self, name, config):
        """Save configuration to file."""
        pass

    def load(self, name):
        """Load configuration from file."""
        pass

    def export_env(self, config):
        """Export as .env format."""
        pass

# rag_web_performance.py
class PerformanceTracker:
    def record_indexing(self, duration, chunks):
        """Record indexing metrics."""
        pass

    def record_query(self, duration, results, cache_hit):
        """Record query metrics."""
        pass

    def get_dashboard_data(self):
        """Get data for performance dashboard."""
        pass
```

---

## 📅 Implementation Timeline

### Week 1: Foundation (Product Manager + UX Researcher + Python Pro)
**Days 1-2:**
- Product Manager: Define all requirements
- UX Researcher: User research and workflow analysis
- Python Pro: Design backend architecture

**Days 3-5:**
- Python Pro: Implement backend managers
- All: Review and alignment

**Deliverables:**
- Requirements document
- UX research findings
- Backend foundation code

### Week 2: Core Features (Frontend + UI Designer + Refactoring Specialist)
**Days 1-2:**
- UI Designer: Create mockups for missing features
- Frontend Developer: Implement parameter controls

**Days 3-4:**
- Frontend Developer: Implement performance dashboard
- Refactoring Specialist: Parameter registry refactor

**Day 5:**
- All: Integration and testing

**Deliverables:**
- All Priority 1 features implemented
- Performance dashboard
- Refactored parameter system

### Week 3: Advanced Features (All Agents)
**Days 1-2:**
- Frontend + Python: Chat mode implementation
- UI Designer: Polish and refinement

**Days 3-4:**
- Frontend: Configuration management
- Refactoring Specialist: Code cleanup

**Day 5:**
- All: Final testing and documentation
- Agent Organizer: Release preparation

**Deliverables:**
- Chat mode complete
- Configuration export/import
- Full documentation
- Release v2.0

---

## ✅ Success Criteria

### Functional Requirements
- ✅ All 30+ CLI parameters available in GUI
- ✅ No feature requires CLI access
- ✅ Configuration export/import works
- ✅ Performance dashboard shows real-time metrics

### Non-Functional Requirements
- ✅ Page load time < 3 seconds
- ✅ Query response time same as CLI
- ✅ No regressions in existing features
- ✅ Mobile-responsive (bonus)

### Quality Gates
- ✅ Unit tests for new components
- ✅ Integration tests for end-to-end flows
- ✅ Accessibility audit (WCAG 2.1 AA)
- ✅ Performance benchmarks (no degradation)

---

## 🚀 Quick Start for Implementation

### For Product Manager:
```bash
# Review and prioritize
cd /Users/frytos/code/llamaIndex-local-rag
cat GUI_FEATURE_PARITY_PLAN.md
# Create issues in GitHub/Jira from user stories
```

### For Developers:
```bash
# Create feature branch
git checkout -b feature/gui-cli-parity

# Start with backend
touch rag_web_backend.py
touch rag_web_components.py
touch rag_web_config.py

# Implement Phase 1 features
# ...

# Test
streamlit run rag_web_enhanced.py

# Commit
git add .
git commit -m "feat: add missing CLI parameters to GUI"
```

### For Designers:
```bash
# Create design files
mkdir -p design/mockups
# Create Figma/Sketch mockups for new features
# Share with team for feedback
```

---

## 📊 Effort Estimation

| Agent | Hours | Days (8h) |
|-------|-------|-----------|
| Product Manager | 16 | 2 |
| UX Researcher | 16 | 2 |
| UI Designer | 24 | 3 |
| Frontend Developer | 56 | 7 |
| Python Pro | 32 | 4 |
| Refactoring Specialist | 24 | 3 |
| Agent Organizer | 12 | 1.5 |
| **Total** | **180** | **22.5** |

**With 3 developers working in parallel: 10-15 days**

---

## 🎯 Next Steps

1. **Review this plan** with team
2. **Assign agents** to tasks
3. **Create detailed tickets** from user stories
4. **Set up project board** (Kanban/Scrum)
5. **Kickoff meeting** with all agents
6. **Begin Sprint 1** (Week 1)

---

This plan provides **complete GUI feature parity** with the CLI, organized by agent responsibilities with detailed specifications and timeline.
