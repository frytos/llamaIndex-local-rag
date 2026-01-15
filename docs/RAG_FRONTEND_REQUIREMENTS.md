# RAG Frontend Requirements Document

**Project:** Custom RAG Configuration Interface
**Version:** 1.0
**Date:** January 15, 2026
**Status:** Requirements Definition

---

## Executive Summary

A modern, user-friendly web interface for configuring and managing RAG (Retrieval-Augmented Generation) pipelines, replacing the current Streamlit-based UI with a production-ready React/Next.js application optimized for the local RAG pipeline.

**Key Goals:**
- Simplify RAG setup from 60-90 minutes to 10-15 minutes
- Provide visual guidance for parameter tuning
- Support multiple deployment targets (local, RunPod, cloud)
- Enable configuration presets for common use cases
- Real-time validation and preview

---

## 1. User Personas

### Persona 1: Technical Researcher
**Name:** Dr. Sarah (PhD candidate)
**Goals:** Index research papers, query academic documents
**Technical Level:** Medium (knows Python basics, not ML expert)
**Pain Points:**
- Overwhelmed by configuration options
- Unsure which embedding model to use
- Doesn't know optimal chunk sizes
**Needs:**
- Guided setup with recommendations
- "Research Papers" preset configuration
- Clear explanations of parameters

### Persona 2: Privacy-Conscious Power User
**Name:** Alex (journalist/activist)
**Goals:** Index chat archives, personal documents, sensitive materials
**Technical Level:** Medium-High (comfortable with terminal, Docker)
**Pain Points:**
- Needs 100% local processing (zero cloud)
- Wants to understand what's happening
- Needs to verify data never leaves machine
**Needs:**
- Local-only mode toggle
- Privacy indicators (green checkmarks)
- Export/backup functionality

### Persona 3: Indie Developer
**Name:** Jamie (building a product)
**Goals:** Embed RAG into their application
**Technical Level:** High (full-stack developer)
**Pain Points:**
- Needs API-first interface
- Wants to automate configuration
- Needs performance monitoring
**Needs:**
- REST API for configuration
- Configuration as code (YAML/JSON)
- Performance metrics dashboard

---

## 2. User Journeys

### Journey 1: First-Time Setup (Target: 10 minutes)

**Current Experience:** 60-90 minutes, 8% success rate

**Desired Experience:**

```
1. Welcome Screen (30 seconds)
   - Choose use case: Chat Archives | Research Papers | Code Documentation | Custom

2. Document Selection (2 minutes)
   - Drag & drop folder or files
   - Preview: "Found 1,247 files, ~850MB"
   - Auto-detect: "Detected Facebook Messenger export (HTML)"

3. Configuration Preset (1 minute)
   - Show recommended settings for detected use case
   - "Chat Archives" preset auto-fills:
     * Chunk size: 300 (fits 3-5 messages)
     * Overlap: 50 (15%)
     * Embedding: bge-m3 (multilingual)
     * Extract metadata: ON
   - Allow customization (collapsed by default)

4. Infrastructure Selection (1 minute)
   - Local (M1 Mac) ✓ Recommended for privacy
   - RunPod GPU (faster, costs ~$0.40/hour)
   - Show estimates: "Indexing will take ~8 minutes locally"

5. Review & Start (30 seconds)
   - Summary card showing all selections
   - Estimated time and cost
   - "Start Indexing" button

6. Progress Monitoring (8 minutes)
   - Real-time progress bar with stages
   - Live metrics: "Embedded 3,247/10,000 chunks (32%)"
   - ETA countdown

7. First Query (1 minute)
   - Auto-opens query interface when complete
   - Example queries suggested
   - Test query: "What did we discuss about..."
```

**Total:** 13.5 minutes (vs 90 minutes current)

### Journey 2: Advanced Configuration

**For power users who want full control:**

```
1. Advanced Mode Toggle
   - Unlock all parameters
   - Show tooltips with explanations

2. Parameter Sections:
   - Document Processing
   - Embedding Configuration
   - Vector Storage
   - Retrieval Settings
   - LLM Configuration
   - Performance Tuning

3. Real-time Validation:
   - Show warnings: "Chunk size 2000 may exceed context window"
   - Show recommendations: "For chat data, we recommend chunk size 300-500"

4. Save as Preset:
   - Name: "My Custom Config"
   - Export as JSON
   - Share with team
```

### Journey 3: Query Interface

```
1. Query Input
   - Natural language question
   - Optional filters: Date range, participant, document type

2. Live Results
   - Streaming response (token by token)
   - Retrieved chunks shown alongside
   - Relevance scores visible

3. Source Attribution
   - Click chunk → view original document context
   - Highlight: Shows exact location in source

4. Query History
   - Previous queries saved
   - One-click re-run
   - Export conversation
```

---

## 3. Feature Requirements

### 3.1 Document Selection & Upload

**Requirements:**

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F1.1 | Drag & Drop Upload | P0 | Drag folder/files onto interface |
| F1.2 | Folder Browser | P0 | Browse file system, select documents |
| F1.3 | File Type Detection | P1 | Auto-detect: PDF, HTML, DOCX, TXT, MD, chat exports |
| F1.4 | Preview Panel | P1 | Show file count, total size, sample content |
| F1.5 | Exclusion Filters | P2 | Exclude file types, patterns (*.log, node_modules/) |
| F1.6 | URL Import | P2 | Paste URL to index website/documentation |
| F1.7 | Cloud Storage | P3 | Connect Google Drive, Dropbox, OneDrive |

**Technical Specs:**
- Maximum file size warning: >100MB per file
- Supported formats: PDF, HTML, DOCX, TXT, MD, RTF, CSV
- Batch processing: Up to 10,000 files
- Progress indicator for file parsing

---

### 3.2 Embedding Model Selection

**Requirements:**

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F2.1 | Model Selector | P0 | Dropdown with recommended models |
| F2.2 | Model Comparison | P1 | Show speed, quality, language support |
| F2.3 | Auto-Recommendation | P1 | Based on document type and language |
| F2.4 | Backend Selection | P1 | HuggingFace vs MLX (M1 only) |
| F2.5 | Custom Model | P2 | Upload/specify custom model |

**Model Options:**

```typescript
interface EmbeddingModel {
  id: string;
  name: string;
  dimensions: number;
  languages: string[];
  speed: 'slow' | 'medium' | 'fast';
  quality: 'good' | 'better' | 'best';
  recommended_for: string[];
  requirements: {
    ram_gb: number;
    gpu_optional: boolean;
    apple_silicon_optimized: boolean;
  };
}

const models: EmbeddingModel[] = [
  {
    id: 'BAAI/bge-small-en',
    name: 'BGE Small (English)',
    dimensions: 384,
    languages: ['English'],
    speed: 'fast',
    quality: 'good',
    recommended_for: ['Quick testing', 'English documents', 'Limited RAM'],
    requirements: { ram_gb: 2, gpu_optional: true, apple_silicon_optimized: true }
  },
  {
    id: 'BAAI/bge-m3',
    name: 'BGE M3 (Multilingual)',
    dimensions: 1024,
    languages: ['100+ languages'],
    speed: 'medium',
    quality: 'best',
    recommended_for: ['Chat archives', 'Multilingual docs', 'Production use'],
    requirements: { ram_gb: 4, gpu_optional: true, apple_silicon_optimized: true }
  },
  {
    id: 'sentence-transformers/all-MiniLM-L6-v2',
    name: 'MiniLM (Lightweight)',
    dimensions: 384,
    languages: ['English'],
    speed: 'fast',
    quality: 'good',
    recommended_for: ['Low-resource systems', 'Fast indexing'],
    requirements: { ram_gb: 1, gpu_optional: false, apple_silicon_optimized: true }
  }
];
```

**UI Components:**
- Model cards with comparison metrics
- Speed/Quality/Size visualization
- "Show me the best model for..." AI assistant
- Backend toggle: "Use MLX on Apple Silicon" (3.7x faster)

---

### 3.3 Infrastructure Selection

**Requirements:**

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F3.1 | Local Deployment | P0 | Run on user's machine (M1/M2/M3 Mac, Linux, Windows) |
| F3.2 | RunPod GPU | P1 | Deploy to RunPod for GPU acceleration |
| F3.3 | Resource Estimation | P1 | Show RAM/GPU/Storage requirements |
| F3.4 | Cost Calculator | P1 | Estimate costs: $0 local, $0.40/hour RunPod |
| F3.5 | Health Checks | P1 | Verify Docker, PostgreSQL, GPU availability |
| F3.6 | One-Click Local Setup | P0 | Auto-start Docker Compose with PostgreSQL |

**Infrastructure Options:**

```typescript
interface InfrastructureOption {
  id: string;
  name: string;
  type: 'local' | 'cloud' | 'hybrid';
  cost_per_hour: number;
  performance_multiplier: number;
  privacy_level: 'private' | 'shared_infra' | 'public';
  requirements: string[];
  auto_setup: boolean;
}

const infrastructures: InfrastructureOption[] = [
  {
    id: 'local-m1',
    name: 'Local (Apple Silicon)',
    type: 'local',
    cost_per_hour: 0,
    performance_multiplier: 1.0,
    privacy_level: 'private',
    requirements: ['16GB RAM', 'macOS 12+', 'Docker Desktop'],
    auto_setup: true
  },
  {
    id: 'runpod-rtx4090',
    name: 'RunPod GPU (RTX 4090)',
    type: 'cloud',
    cost_per_hour: 0.40,
    performance_multiplier: 15.0,
    privacy_level: 'shared_infra',
    requirements: ['RunPod API key', 'Internet connection'],
    auto_setup: true
  },
  {
    id: 'hybrid',
    name: 'Hybrid (Local + Cloud LLM)',
    type: 'hybrid',
    cost_per_hour: 0.10,
    performance_multiplier: 3.0,
    privacy_level: 'private',
    requirements: ['Local PostgreSQL', 'vLLM server'],
    auto_setup: false
  }
];
```

**UI Components:**
- Infrastructure comparison cards
- Privacy indicator (green shield for local)
- Cost calculator: "Indexing 10,000 chunks will cost: $0 (local) or $1.20 (RunPod)"
- Resource availability checker (Docker running? PostgreSQL accessible?)

---

### 3.4 LLM Selection

**Requirements:**

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F4.1 | LLM Model Selector | P0 | Choose: Mistral 7B, Llama 3, Custom |
| F4.2 | Backend Selection | P0 | llama.cpp (local) vs vLLM (GPU) |
| F4.3 | Quantization Options | P1 | Q4, Q5, Q8, FP16 with quality/speed tradeoff |
| F4.4 | Context Window Config | P1 | 2048, 4096, 8192, 32768 tokens |
| F4.5 | Temperature Control | P1 | Slider 0.0-1.0 with explanations |
| F4.6 | Model Download | P1 | Auto-download with progress bar |

**LLM Options:**

```typescript
interface LLMOption {
  id: string;
  name: string;
  size_gb: number;
  context_window: number;
  backend: 'llama.cpp' | 'vllm' | 'both';
  quantization: string[];
  speed_tokens_per_sec: {
    cpu: number;
    metal: number;
    cuda: number;
  };
  recommended_for: string[];
}

const llms: LLMOption[] = [
  {
    id: 'mistral-7b-instruct',
    name: 'Mistral 7B Instruct v0.3',
    size_gb: 4.2,
    context_window: 32768,
    backend: 'both',
    quantization: ['Q4_K_M', 'Q5_K_M', 'Q8_0', 'FP16'],
    speed_tokens_per_sec: { cpu: 15, metal: 25, cuda: 80 },
    recommended_for: ['General Q&A', 'Document analysis', 'Chat archives']
  },
  {
    id: 'llama-3.1-8b',
    name: 'Llama 3.1 8B',
    size_gb: 4.7,
    context_window: 128000,
    backend: 'both',
    quantization: ['Q4_K_M', 'Q5_K_M', 'Q8_0'],
    speed_tokens_per_sec: { cpu: 12, metal: 20, cuda: 70 },
    recommended_for: ['Long context', 'Code analysis', 'Research']
  }
];
```

**UI Components:**
- Model cards with download status
- Speed/Quality tradeoff visualizer
- Context window calculator: "Your average chunk is 700 chars, fits X chunks in context"
- Backend auto-detection: "Metal GPU detected, vLLM recommended"

---

### 3.5 Parameter Configuration

**Requirements:**

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F5.1 | Chunking Parameters | P0 | Size, overlap with visual preview |
| F5.2 | Retrieval Settings | P0 | Top-K, similarity threshold, hybrid alpha |
| F5.3 | Performance Tuning | P1 | Batch sizes, GPU layers, context window |
| F5.4 | Advanced Options | P2 | Reranking, query expansion, semantic cache |
| F5.5 | Parameter Validation | P0 | Real-time validation with warnings |
| F5.6 | Smart Defaults | P1 | AI-suggested defaults based on use case |

**Configuration Schema:**

```typescript
interface RAGConfiguration {
  // Document Processing
  document_processing: {
    chunk_size: number;           // 100-2000, default: 700
    chunk_overlap: number;        // 0-500, default: 150
    extract_metadata: boolean;    // default: true
    clean_html: boolean;          // default: true for HTML
  };

  // Embedding
  embedding: {
    model: string;                // Model ID
    backend: 'huggingface' | 'mlx';
    dimensions: number;           // Auto-set based on model
    batch_size: number;           // 32-256, default: 64
  };

  // Vector Storage
  vector_storage: {
    table_name: string;           // Auto-generated or custom
    reset_existing: boolean;      // Drop table if exists
    enable_hnsw: boolean;         // default: true (100x+ speedup)
    hnsw_ef_construction: number; // 64-512, default: 128
  };

  // Retrieval
  retrieval: {
    top_k: number;                // 1-10, default: 4
    similarity_threshold: number;  // 0.0-1.0, default: 0.7
    enable_reranking: boolean;    // default: false
    enable_hybrid: boolean;       // Vector + BM25, default: false
    hybrid_alpha: number;         // 0.0-1.0, default: 0.5
  };

  // LLM
  llm: {
    model: string;                // Model ID
    backend: 'llamacpp' | 'vllm';
    context_window: number;       // 2048-32768
    max_new_tokens: number;       // 128-512, default: 256
    temperature: number;          // 0.0-1.0, default: 0.1
    n_gpu_layers: number;         // 0-32, default: 24 for M1
  };

  // Advanced
  advanced: {
    enable_semantic_cache: boolean;  // default: true
    cache_threshold: number;         // 0.0-1.0, default: 0.92
    enable_query_expansion: boolean; // default: false
    enable_conversation_memory: boolean; // default: false
  };

  // Infrastructure
  infrastructure: {
    deployment: 'local' | 'runpod' | 'custom';
    database_host: string;
    database_port: number;
    use_gpu: boolean;
  };
}
```

**UI Components:**
- Tabbed interface: Basic | Advanced | Expert
- Smart sliders with recommended ranges (green zone)
- Visual chunk preview: "Your chunks will look like this..."
- Parameter impact calculator: "Increasing chunk size → longer indexing, better context"
- Conflict detection: "Warning: Top-K=10 + Chunk Size=2000 may exceed context window"

---

## 4. Core Features Specification

### 4.1 Configuration Wizard (P0)

**Component:** `ConfigurationWizard.tsx`

**Features:**
- Step-by-step guided setup (6 steps)
- Progress indicator at top
- "Back" and "Next" navigation
- "Skip wizard, go to advanced" option
- Save/Resume configuration (localStorage)

**Steps:**
1. Use Case Selection
2. Document Upload
3. Preset Configuration
4. Infrastructure Choice
5. Review Summary
6. Start Indexing

**Validation:**
- Each step validates before allowing "Next"
- Clear error messages with solutions
- "Why is this needed?" help tooltips

**Responsive Design:**
- Mobile-friendly (collapsible sections)
- Tablet-optimized (2-column layout)
- Desktop (side-by-side preview)

---

### 4.2 Parameter Configuration Panel (P0)

**Component:** `ParameterPanel.tsx`

**Layout:**

```
┌─────────────────────────────────────────────────────┐
│ Configuration Preset: [Chat Archives ▼]  [Save As] │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 📄 Document Processing                             │
│ ├─ Chunk Size:        [━━━●━━] 700 chars          │
│ │  └─ Preview: "Each chunk fits ~3-5 messages"    │
│ ├─ Chunk Overlap:     [━●━━━━] 150 chars (21%)    │
│ └─ Extract Metadata:  [✓] ON                       │
│                                                     │
│ 🧬 Embedding Model                                 │
│ ├─ Model: [BGE M3 (Multilingual) ▼]               │
│ │  └─ 1024 dimensions, 100+ languages              │
│ ├─ Backend: [●] MLX   [ ] HuggingFace              │
│ │  └─ 3.7x faster on Apple Silicon                 │
│ └─ Batch Size:        [━━━●━━] 128                 │
│                                                     │
│ 🔍 Retrieval Settings                              │
│ ├─ Top-K Results:     [━━●━━━] 4 chunks            │
│ ├─ Enable Reranking:  [ ] OFF (15-30% better)      │
│ └─ Enable Hybrid:     [ ] OFF (Vector + BM25)      │
│                                                     │
│ 🤖 LLM Configuration                               │
│ ├─ Model: [Mistral 7B Instruct ▼]                 │
│ ├─ Backend: [●] vLLM  [ ] llama.cpp               │
│ │  └─ 3-4x faster generation                       │
│ ├─ Context Window:    [━━━━●━] 8192 tokens        │
│ └─ Temperature:       [●━━━━━] 0.1 (focused)       │
│                                                     │
│ [Show Advanced Settings ▼]                         │
└─────────────────────────────────────────────────────┘
```

**Features:**
- Real-time preview of changes
- Undo/Redo support
- Reset to defaults button
- Export/Import JSON configuration
- Validation warnings inline

---

### 4.3 Visual Chunk Preview (P1)

**Component:** `ChunkPreview.tsx`

**Features:**
- Live preview of how documents will be chunked
- Sample document visualization
- Overlap highlighting (show shared text between chunks)
- Character/token count per chunk
- "Test with your document" uploader

**Example Preview:**

```
┌────────────────────────────────────────┐
│ Chunk 1 (698 chars)                    │
│ ┌────────────────────────────────────┐ │
│ │ The quick brown fox jumps over     │ │
│ │ the lazy dog. This is a sample     │ │
│ │ document that demonstrates how     │ │
│ │ chunking works with overlap...     │ │
│ │ [continues...]                      │ │
│ └────────────────────────────────────┘ │
│                                        │
│ Overlap (150 chars): ████████          │
│                                        │
│ Chunk 2 (702 chars)                    │
│ ┌────────────────────────────────────┐ │
│ │ ...chunking works with overlap.    │ │  ← Shared text
│ │ The next section introduces new    │ │
│ │ concepts about semantic search...  │ │
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘

Quality Indicators:
✅ Average chunk: 700 chars (good)
✅ Overlap ratio: 21% (recommended)
⚠️  Some chunks exceed 1000 chars
```

---

### 4.4 Indexing Progress Monitor (P0)

**Component:** `IndexingProgressMonitor.tsx`

**Features:**
- Multi-stage progress visualization
- Live metrics dashboard
- Pause/Resume support (checkpoint/resume)
- Error handling with retry options
- ETA calculation

**Progress Stages:**

```
Stage 1: Loading Documents       [████████████░░░░░░░░] 65%
  ↳ Loaded 812/1,247 files
  ↳ Speed: 25 files/sec
  ↳ ETA: 17 seconds

Stage 2: Chunking                [████████████████░░░░] 80%
  ↳ Created 8,234/10,000 chunks
  ↳ Speed: 166 chunks/sec

Stage 3: Embedding (MLX)          [██████░░░░░░░░░░░░░░] 30%
  ↳ Embedded 3,000/10,000 chunks
  ↳ Speed: 93 chunks/sec (3.7x faster!)
  ↳ ETA: 1 minute 15 seconds

Stage 4: Storing in PostgreSQL    [░░░░░░░░░░░░░░░░░░░░] 0%
  ↳ Waiting for embedding to complete...

Stage 5: Building HNSW Index      [░░░░░░░░░░░░░░░░░░░░] 0%
  ↳ Estimated: 15 seconds
```

**Live Metrics Panel:**
```
┌─────────────────────────────────────┐
│ Live Performance Metrics            │
├─────────────────────────────────────┤
│ Throughput:  93 chunks/sec          │
│ Memory:      7.2GB / 16GB (45%)     │
│ GPU:         Metal 85% utilized     │
│ Estimated:   5 minutes 30 seconds   │
└─────────────────────────────────────┘
```

**Error Handling:**
```
❌ Stage 3 failed: Embedding batch 47/156 failed

[Retry] [Skip Batch] [View Error Details]

Error: Out of memory
Suggestion: Reduce EMBED_BATCH from 128 to 64
[Apply Fix & Retry]
```

---

### 4.5 Configuration Presets (P1)

**Component:** `ConfigurationPresets.tsx`

**Built-in Presets:**

```typescript
const presets: Record<string, Partial<RAGConfiguration>> = {
  'chat-archives': {
    document_processing: {
      chunk_size: 300,
      chunk_overlap: 50,
      extract_metadata: true,
    },
    embedding: {
      model: 'BAAI/bge-m3',
      backend: 'mlx',
      dimensions: 1024,
      batch_size: 128,
    },
    retrieval: {
      top_k: 6,
      similarity_threshold: 0.7,
    },
    description: 'Optimized for Messenger, Slack, Discord archives',
    icon: '💬',
  },

  'research-papers': {
    document_processing: {
      chunk_size: 800,
      chunk_overlap: 200,
      extract_metadata: true,
    },
    embedding: {
      model: 'BAAI/bge-large-en-v1.5',
      dimensions: 1024,
      batch_size: 64,
    },
    retrieval: {
      top_k: 4,
      similarity_threshold: 0.75,
      enable_reranking: true,
    },
    description: 'Optimized for academic papers, citations, technical docs',
    icon: '📚',
  },

  'code-documentation': {
    document_processing: {
      chunk_size: 500,
      chunk_overlap: 100,
    },
    embedding: {
      model: 'BAAI/bge-small-en',
      dimensions: 384,
      batch_size: 128,
    },
    retrieval: {
      top_k: 5,
      enable_hybrid: true,
      hybrid_alpha: 0.6,
    },
    description: 'Optimized for code repos, API docs, README files',
    icon: '👨‍💻',
  },

  'fast-testing': {
    document_processing: {
      chunk_size: 500,
      chunk_overlap: 100,
    },
    embedding: {
      model: 'sentence-transformers/all-MiniLM-L6-v2',
      backend: 'mlx',
      dimensions: 384,
      batch_size: 256,
    },
    llm: {
      backend: 'vllm',
      n_gpu_layers: 32,
    },
    description: 'Maximum speed for testing and iteration',
    icon: '⚡',
  },
};
```

**UI Design:**

```
┌──────────────────────────────────────────────────┐
│ Choose a Preset or Start from Scratch           │
├──────────────────────────────────────────────────┤
│                                                  │
│  [💬 Chat Archives]  [📚 Research Papers]       │
│   Messenger, Slack      Academic papers          │
│   Chunk: 300            Chunk: 800               │
│   Model: bge-m3         Model: bge-large         │
│                                                  │
│  [👨‍💻 Code Docs]      [⚡ Fast Testing]         │
│   GitHub repos          Quick iteration          │
│   Chunk: 500            Chunk: 500               │
│   Hybrid search         MiniLM + vLLM            │
│                                                  │
│  [🛠️ Custom Configuration]                      │
│   Configure everything manually                  │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

### 4.6 Query Interface (P0)

**Component:** `QueryInterface.tsx`

**Features:**

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F6.1 | Natural Language Query | P0 | Text input with autocomplete |
| F6.2 | Streaming Responses | P0 | Token-by-token display with SSE |
| F6.3 | Source Attribution | P0 | Show retrieved chunks with scores |
| F6.4 | Metadata Filters | P1 | Filter by date, participant, document |
| F6.5 | Query History | P1 | Save/recall previous queries |
| F6.6 | Export Results | P2 | PDF, Markdown, Copy to clipboard |
| F6.7 | Query Suggestions | P2 | AI-suggested follow-up questions |

**UI Layout:**

```
┌─────────────────────────────────────────────────────┐
│ 🔍 Ask a question about your documents...          │
│ [What did Alice say about the project?          ] │
│                                        [Ask] [Clear]│
├─────────────────────────────────────────────────────┤
│                                                     │
│ 💬 Answer (Streaming...)                           │
│ ┌─────────────────────────────────────────────────┐│
│ │ Based on the conversations, Alice mentioned     ││
│ │ that the project deadline was extended to...█   ││
│ │                                                  ││
│ │ Sources: 4 relevant chunks found                ││
│ └─────────────────────────────────────────────────┘│
│                                                     │
│ 📑 Retrieved Sources                               │
│ ┌─────────────────────────────────────────────────┐│
│ │ Chunk 1 (Score: 0.87) - 2024-03-15             ││
│ │ 📱 Messenger: Alice → You                       ││
│ │ "The deadline has been pushed to next Friday    ││
│ │  because we need more time for testing..."      ││
│ │                                    [View Full] ││
│ ├─────────────────────────────────────────────────┤│
│ │ Chunk 2 (Score: 0.82) - 2024-03-14             ││
│ │ 📱 Messenger: Alice → Team                      ││
│ │ [Preview text...]                  [View Full] ││
│ └─────────────────────────────────────────────────┘│
│                                                     │
│ ⏱️ Query took 2.3 seconds                          │
│    └─ Retrieval: 2ms | Generation: 2.3s           │
│                                                     │
│ [Export Answer] [Share] [Save to History]          │
└─────────────────────────────────────────────────────┘

Sidebar:
┌──────────────────┐
│ 📊 Filters       │
├──────────────────┤
│ Date Range:      │
│ [2024-01 to Now] │
│                  │
│ Participant:     │
│ [All ▼]          │
│                  │
│ Document Type:   │
│ [All ▼]          │
│                  │
│ [Apply Filters]  │
└──────────────────┘

Bottom:
┌──────────────────┐
│ 📜 History       │
├──────────────────┤
│ • What did...    │
│ • How to...      │
│ • When was...    │
│   [Clear History]│
└──────────────────┘
```

---

### 4.7 Dashboard & Monitoring (P1)

**Component:** `RAGDashboard.tsx`

**Metrics Display:**

```
┌─────────────────────────────────────────────────────┐
│ RAG Pipeline Status                                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│ [Active] fb_messenger_archive    91,247 chunks     │
│          Last updated: 2 hours ago                  │
│          Queries today: 127                         │
│                                                     │
│ Performance Metrics (Last 24h)                      │
│ ┌────────────┬────────────┬────────────┬──────────┐│
│ │ Avg Latency│ Cache Hits │ Queries    │ Errors   ││
│ │   2.3s     │    42%     │    127     │    0     ││
│ └────────────┴────────────┴────────────┴──────────┘│
│                                                     │
│ Query Latency (p95): [━━━━━●━━━━] 2.1s / 5.0s     │
│ Memory Usage:        [━━━━━━●━━━] 11.2GB / 16GB   │
│ Cache Hit Rate:      [━━━━●━━━━━] 42% / 60%       │
│                                                     │
│ [View Detailed Metrics] [Export Report]            │
└─────────────────────────────────────────────────────┘
```

**Features:**
- Real-time metric updates (WebSocket)
- Historical trend charts (Chart.js/Recharts)
- Alert indicators (red/yellow/green)
- Quick actions: Clear cache, Rebuild index, Export data

---

### 4.8 Index Management (P1)

**Component:** `IndexManager.tsx`

**Features:**

| Feature | Description |
|---------|-------------|
| List Indexes | Show all vector tables with metadata |
| Delete Index | Soft delete with confirmation dialog |
| Duplicate Index | Clone configuration for testing |
| Update Index | Add new documents incrementally |
| Export Index | Download embeddings, metadata as JSON/CSV |
| Index Stats | Document count, size, last updated, query count |

**UI:**

```
┌─────────────────────────────────────────────────────┐
│ Your Vector Indexes                                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 📚 fb_messenger_archive                            │
│ │  91,247 chunks | 285MB | Last query: 5 min ago  │
│ │  Model: bge-m3 (1024-dim) | Created: Jan 10      │
│ │  [Query] [Update] [Export] [Delete]              │
│                                                     │
│ 📄 research_papers_2024                            │
│ │  15,623 chunks | 92MB | Last query: 2 days ago  │
│ │  Model: bge-large (1024-dim) | Created: Dec 15   │
│ │  [Query] [Update] [Export] [Delete]              │
│                                                     │
│ [+ Create New Index]                                │
└─────────────────────────────────────────────────────┘
```

---

## 5. Technical Requirements

### 5.1 Frontend Stack

**Recommended Technology:**

```typescript
// Core Framework
Framework: Next.js 14+ (App Router)
Language: TypeScript (strict mode)
Styling: Tailwind CSS + shadcn/ui components
State: Zustand (lightweight, performant)

// UI Components
Component Library: shadcn/ui (accessible, customizable)
Icons: Lucide React
Charts: Recharts or Chart.js
Forms: React Hook Form + Zod validation

// Data Fetching
API Client: Axios or Fetch with React Query
Real-time: Server-Sent Events (SSE) for streaming
WebSocket: For live metrics (optional)

// Build & Development
Build: Next.js compiler (Turbopack)
Linting: ESLint + Prettier
Type Checking: TypeScript strict mode
Testing: Vitest + React Testing Library
```

**Why This Stack:**
- **Next.js:** SSR/SSG for SEO, App Router for modern patterns, built-in API routes
- **TypeScript:** Type safety for RAG configurations (prevent runtime errors)
- **Tailwind + shadcn:** Rapid development with accessible components
- **Zustand:** Simple state management (no Redux boilerplate)
- **React Query:** Automatic caching, background refetching, optimistic updates

### 5.2 Backend API Requirements

**API Endpoints Needed:**

```typescript
// Configuration
POST   /api/config/validate      // Validate configuration
POST   /api/config/presets       // Get recommended presets
POST   /api/config/export        // Export as JSON/YAML

// Indexing
POST   /api/index/start          // Start indexing with config
GET    /api/index/status/:id     // Get indexing progress (SSE)
POST   /api/index/pause/:id      // Pause indexing (checkpoint)
POST   /api/index/resume/:id     // Resume from checkpoint
DELETE /api/index/cancel/:id     // Cancel indexing

// Querying
POST   /api/query                // Execute query
GET    /api/query/stream         // Streaming query (SSE)
GET    /api/query/history        // Get query history
POST   /api/query/export         // Export conversation

// Index Management
GET    /api/indexes              // List all indexes
GET    /api/indexes/:name        // Get index details
DELETE /api/indexes/:name        // Delete index
POST   /api/indexes/:name/update // Add documents to existing index

// System
GET    /api/system/health        // Health check
GET    /api/system/resources     // RAM, GPU, disk usage
GET    /api/system/models        // Available models
GET    /api/system/capabilities  // Detect: MLX support, GPU, Docker

// Metrics
GET    /api/metrics              // Prometheus metrics
GET    /api/metrics/dashboard    // Dashboard data (JSON)
```

**FastAPI Backend Structure:**

```python
# api/
├── main.py                  # FastAPI app
├── routers/
│   ├── config.py            # Configuration endpoints
│   ├── indexing.py          # Indexing endpoints
│   ├── query.py             # Query endpoints
│   ├── indexes.py           # Index management
│   └── system.py            # System info
├── models/
│   ├── config.py            # Pydantic models for configuration
│   ├── query.py             # Query request/response models
│   └── index.py             # Index metadata models
├── services/
│   ├── rag_service.py       # Wrap existing RAG pipeline
│   ├── indexing_service.py  # Background indexing with checkpoints
│   └── monitoring_service.py # Metrics collection
└── middleware/
    ├── auth.py              # JWT authentication
    └── rate_limit.py        # Rate limiting
```

### 5.3 State Management

**Zustand Store Structure:**

```typescript
// stores/useConfigStore.ts
interface ConfigStore {
  // Configuration state
  config: RAGConfiguration;
  preset: string | null;
  isDirty: boolean;

  // Actions
  setConfig: (config: Partial<RAGConfiguration>) => void;
  loadPreset: (presetName: string) => void;
  resetToDefaults: () => void;
  validateConfig: () => Promise<ValidationResult>;
  exportConfig: () => string; // JSON
  importConfig: (json: string) => void;
}

// stores/useIndexingStore.ts
interface IndexingStore {
  // Indexing state
  activeJobs: IndexingJob[];
  progress: Record<string, IndexingProgress>;

  // Actions
  startIndexing: (config: RAGConfiguration) => Promise<string>; // job ID
  pauseIndexing: (jobId: string) => Promise<void>;
  resumeIndexing: (jobId: string) => Promise<void>;
  cancelIndexing: (jobId: string) => Promise<void>;

  // Real-time updates via SSE
  subscribeToProgress: (jobId: string) => void;
}

// stores/useQueryStore.ts
interface QueryStore {
  // Query state
  currentQuery: string;
  history: QueryHistoryItem[];
  results: QueryResult | null;
  isStreaming: boolean;

  // Actions
  executeQuery: (query: string, filters?: QueryFilters) => Promise<void>;
  streamQuery: (query: string) => AsyncGenerator<string>;
  clearHistory: () => void;
  exportResults: (format: 'pdf' | 'markdown' | 'json') => void;
}

// stores/useSystemStore.ts
interface SystemStore {
  // System state
  health: SystemHealth;
  resources: ResourceUsage;
  capabilities: SystemCapabilities;

  // Actions
  checkHealth: () => Promise<void>;
  refreshResources: () => Promise<void>;
  detectCapabilities: () => Promise<void>;
}
```

### 5.4 Form Validation

**Validation Rules (Zod):**

```typescript
import { z } from 'zod';

const DocumentProcessingSchema = z.object({
  chunk_size: z.number()
    .min(100, 'Chunk size too small (min: 100)')
    .max(2000, 'Chunk size too large (max: 2000)')
    .refine(val => val >= 300 || confirm('Small chunks may lose context'), {
      message: 'Recommended minimum: 300 for better context'
    }),

  chunk_overlap: z.number()
    .min(0)
    .max(500)
    .refine((val, ctx) => {
      const chunkSize = ctx.parent.chunk_size;
      return val < chunkSize * 0.5;
    }, 'Overlap should be < 50% of chunk size'),

  extract_metadata: z.boolean(),
  clean_html: z.boolean(),
});

const EmbeddingSchema = z.object({
  model: z.enum([
    'BAAI/bge-small-en',
    'BAAI/bge-m3',
    'sentence-transformers/all-MiniLM-L6-v2',
  ]),
  backend: z.enum(['huggingface', 'mlx']),
  dimensions: z.number().int().positive(),
  batch_size: z.number()
    .min(16, 'Batch size too small')
    .max(256, 'Batch size too large (OOM risk)')
    .refine(val => [32, 64, 128, 256].includes(val), {
      message: 'Recommended: 32, 64, 128, or 256'
    }),
});

const RAGConfigSchema = z.object({
  document_processing: DocumentProcessingSchema,
  embedding: EmbeddingSchema,
  vector_storage: VectorStorageSchema,
  retrieval: RetrievalSchema,
  llm: LLMSchema,
  advanced: AdvancedSchema,
  infrastructure: InfrastructureSchema,
}).superRefine((data, ctx) => {
  // Cross-field validation
  const totalChunkTokens = data.document_processing.chunk_size / 4; // rough estimate
  const maxContextTokens = data.llm.context_window;
  const retrievedTokens = totalChunkTokens * data.retrieval.top_k;

  if (retrievedTokens > maxContextTokens * 0.8) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: `Context window overflow risk: ${retrievedTokens} tokens (chunks) > ${maxContextTokens * 0.8} (80% of context window)`,
      path: ['retrieval', 'top_k'],
    });
  }
});
```

---

## 6. UI/UX Specifications

### 6.1 Design System

**Color Palette (Dark Mode):**

```css
/* Primary Colors */
--color-bg-primary: #0d1117;
--color-bg-secondary: #161b22;
--color-bg-tertiary: #21262d;
--color-border: rgba(255, 255, 255, 0.1);

/* Text */
--color-text-primary: #c9d1d9;
--color-text-secondary: #8b949e;
--color-text-muted: #6e7681;

/* Accent Colors */
--color-accent-primary: #58a6ff;    /* Links, actions */
--color-accent-success: #3fb950;    /* Success states */
--color-accent-warning: #d29922;    /* Warnings */
--color-accent-danger: #f85149;     /* Errors, critical */
--color-accent-info: #79c0ff;       /* Info messages */

/* Gradients */
--gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
--gradient-success: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
```

**Typography:**
- Font Family: Inter, -apple-system, BlinkMacSystemFont, sans-serif
- Headings: 700 weight
- Body: 400 weight
- Code: JetBrains Mono, monospace

**Spacing:**
- Base unit: 4px
- Standard spacing: 8px, 16px, 24px, 32px, 48px
- Container max-width: 1400px

### 6.2 Responsive Breakpoints

```css
/* Mobile First */
sm: 640px   /* Mobile landscape, small tablets */
md: 768px   /* Tablets */
lg: 1024px  /* Desktop */
xl: 1280px  /* Large desktop */
2xl: 1536px /* Extra large */
```

**Layout Adaptations:**
- **Mobile (<640px):** Single column, stacked sections, bottom navigation
- **Tablet (640-1024px):** 2-column layout, collapsible sidebar
- **Desktop (>1024px):** 3-column layout, persistent sidebar, side-by-side preview

### 6.3 Accessibility Requirements

**WCAG 2.1 Level AA Compliance:**

| Requirement | Implementation |
|-------------|----------------|
| Keyboard Navigation | Tab order, focus indicators, keyboard shortcuts |
| Screen Reader | ARIA labels, roles, live regions for streaming |
| Color Contrast | 4.5:1 minimum for text, 3:1 for UI components |
| Focus Management | Visible focus rings, skip links, modal trap |
| Error Messages | Clear, actionable, associated with inputs |
| Loading States | Skeleton screens, progress indicators, status announcements |

**Keyboard Shortcuts:**
- `Cmd/Ctrl + K`: Focus query input
- `Cmd/Ctrl + Enter`: Submit query
- `Cmd/Ctrl + ,`: Open settings
- `Esc`: Close modals/dialogs
- `Tab/Shift+Tab`: Navigate between sections

---

## 7. Performance Requirements

### 7.1 Page Load Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| First Contentful Paint (FCP) | <1.5s | Lighthouse |
| Largest Contentful Paint (LCP) | <2.5s | Lighthouse |
| Time to Interactive (TTI) | <3.5s | Lighthouse |
| Cumulative Layout Shift (CLS) | <0.1 | Lighthouse |
| Bundle Size (Initial) | <200KB | webpack-bundle-analyzer |

**Optimization Strategies:**
- Code splitting by route
- Lazy loading for heavy components (charts, editors)
- Image optimization (next/image)
- Font optimization (next/font)
- Tree-shaking unused code

### 7.2 Runtime Performance

| Operation | Target | Validation |
|-----------|--------|------------|
| Configuration Change | <100ms | React DevTools Profiler |
| Form Validation | <50ms | Real-time, non-blocking |
| Query Submission | <200ms | Until SSE stream starts |
| Streaming Token Display | <16ms | 60 FPS smooth rendering |
| Chart Rendering | <500ms | Complex metrics dashboard |

---

## 8. Integration Requirements

### 8.1 Existing System Integration

**Current System:**
- Main pipeline: `rag_low_level_m1_16gb_verbose.py` (3,277 lines)
- Web UI: `rag_web.py` (Streamlit, 2,085 lines)
- Interactive CLI: `rag_interactive.py`

**Integration Strategy:**

**Option A: Wrap Existing Code (Faster - 2-3 weeks)**
```python
# Create FastAPI wrapper around existing functions
from rag_low_level_m1_16gb_verbose import (
    load_documents,
    chunk_documents,
    build_embed_model,
    embed_nodes,
    make_vector_store,
    build_llm,
    run_query,
)

@app.post("/api/index/start")
async def start_indexing(config: RAGConfig):
    # Wrap existing functions with async/progress tracking
    job_id = generate_job_id()

    # Run in background with progress updates
    background_tasks.add_task(
        index_documents_with_progress,
        job_id, config
    )

    return {"job_id": job_id, "status": "started"}
```

**Option B: Refactor First (Better - 6-8 weeks)**
- Refactor monolithic file into modules
- Implement proper service layer
- Add dependency injection
- Build API on top of clean architecture

**Recommendation:** Start with Option A, migrate to Option B over time

### 8.2 Database Integration

**Requirements:**
- Connect to existing PostgreSQL + pgvector
- Support multiple vector tables (indexes)
- Read-only queries for safety
- Connection pooling (avoid the 50-100ms overhead)

**API Models:**

```python
from pydantic import BaseModel

class VectorIndex(BaseModel):
    table_name: str
    document_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int
    created_at: datetime
    last_updated: datetime
    size_mb: float
    query_count_24h: int
    avg_latency_ms: float
```

### 8.3 Real-Time Updates

**Server-Sent Events (SSE):**

```typescript
// Frontend
const useIndexingProgress = (jobId: string) => {
  const [progress, setProgress] = useState<IndexingProgress | null>(null);

  useEffect(() => {
    const eventSource = new EventSource(`/api/index/status/${jobId}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data);
    };

    eventSource.onerror = () => {
      eventSource.close();
    };

    return () => eventSource.close();
  }, [jobId]);

  return progress;
};

// Backend
@app.get("/api/index/status/{job_id}")
async def indexing_status(job_id: str):
    async def event_generator():
        while True:
            progress = get_indexing_progress(job_id)

            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": progress.stage,
                    "current": progress.current,
                    "total": progress.total,
                    "percentage": progress.percentage,
                    "eta_seconds": progress.eta_seconds,
                    "metrics": progress.metrics,
                })
            }

            if progress.status in ['completed', 'failed']:
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())
```

---

## 9. Security Requirements

### 9.1 Authentication (P0)

**Requirements:**

| Feature | Priority | Implementation |
|---------|----------|----------------|
| User Login | P0 | Email + password, OAuth (GitHub, Google) |
| Session Management | P0 | JWT tokens, 30-minute timeout |
| Role-Based Access | P1 | Admin, Editor, Viewer roles |
| API Key Auth | P1 | For programmatic access |
| Multi-Factor Auth | P2 | TOTP, SMS (enterprise) |

**User Roles:**

```typescript
enum Role {
  ADMIN = 'admin',     // Full access: create/delete indexes, manage users
  EDITOR = 'editor',   // Create indexes, query, cannot delete
  VIEWER = 'viewer',   // Query only, read-only
}

interface User {
  id: string;
  email: string;
  role: Role;
  created_at: string;
  last_login: string;
  indexes_owned: string[];  // Can only delete own indexes
}
```

**Protected Routes:**
- `/` → Redirect to `/login` if not authenticated
- `/dashboard` → Requires authentication
- `/admin` → Requires admin role
- `/api/*` → Requires valid JWT token

### 9.2 Data Privacy

**Requirements:**
- ❌ No telemetry/analytics (privacy-first)
- ✅ All processing local (no external API calls)
- ✅ Clear privacy indicators throughout UI
- ✅ "Privacy Mode" toggle (disables any optional external calls)
- ✅ Session data encrypted in localStorage
- ✅ Clear session data on logout

**Privacy Indicators:**

```tsx
<PrivacyBadge
  status="local"  // Green shield
  tooltip="100% local processing. Your data never leaves this machine."
/>

<PrivacyBadge
  status="cloud"  // Yellow shield
  tooltip="Using RunPod GPU. Data transmitted over encrypted connection."
/>
```

---

## 10. Configuration Presets Deep-Dive

### 10.1 Preset: Chat Archives

**Optimized for:** Facebook Messenger, Slack, Discord, WhatsApp

```yaml
name: Chat Archives
icon: 💬
description: Optimized for conversational data with metadata extraction

document_processing:
  chunk_size: 300              # Fits 3-5 messages per chunk
  chunk_overlap: 50            # 15% overlap
  extract_metadata: true       # Participant, date, thread info
  clean_html: true             # Remove Messenger HTML artifacts

embedding:
  model: BAAI/bge-m3           # Multilingual (100+ languages)
  backend: mlx                 # 3.7x faster on Apple Silicon
  dimensions: 1024
  batch_size: 128              # Balance speed/memory

vector_storage:
  enable_hnsw: true            # 100x+ speedup
  hnsw_ef_construction: 128

retrieval:
  top_k: 6                     # More context for conversations
  similarity_threshold: 0.7
  enable_reranking: true       # 15-30% better relevance

llm:
  model: mistral-7b-instruct
  backend: vllm                # 3-4x faster if available
  context_window: 8192
  temperature: 0.1             # Focused answers

advanced:
  enable_semantic_cache: true  # 10,000x for similar queries
  cache_threshold: 0.92
  enable_query_expansion: false
  enable_conversation_memory: true

estimated_performance:
  indexing_time: "8-12 minutes for 10,000 messages"
  query_latency: "2-3 seconds with vLLM"
  memory_required: "8-10GB RAM"

recommended_hardware:
  - "M1/M2/M3 Mac with 16GB RAM"
  - "RunPod GPU for >100K messages"
```

### 10.2 Preset: Research Papers

**Optimized for:** Academic papers, technical documentation, books

```yaml
name: Research Papers
icon: 📚
description: Optimized for long-form academic content with citations

document_processing:
  chunk_size: 800              # Preserve paragraphs/sections
  chunk_overlap: 200           # 25% overlap for context
  extract_metadata: true       # Title, author, page, citations

embedding:
  model: BAAI/bge-large-en-v1.5
  dimensions: 1024
  batch_size: 64               # Balance quality/speed

retrieval:
  top_k: 4                     # Focused retrieval
  similarity_threshold: 0.75   # Higher precision
  enable_reranking: true       # Critical for academic accuracy
  enable_hybrid: true          # Vector + BM25 for technical terms
  hybrid_alpha: 0.7            # Favor vector search

llm:
  context_window: 16384        # Longer context for papers
  temperature: 0.0             # Factual, precise answers
  max_new_tokens: 512          # Longer explanations
```

### 10.3 Preset: Code Documentation

**Optimized for:** GitHub repos, API docs, README files

```yaml
name: Code Documentation
icon: 👨‍💻
description: Optimized for code, API references, technical docs

document_processing:
  chunk_size: 500              # Function-level granularity
  chunk_overlap: 100
  code_aware_splitting: true   # Don't split mid-function

embedding:
  model: BAAI/bge-small-en
  dimensions: 384
  batch_size: 128              # Fast indexing

retrieval:
  top_k: 5
  enable_hybrid: true          # Keyword match for function names
  hybrid_alpha: 0.6            # Balance vector + keyword

llm:
  model: codellama-7b          # Code-specialized model
  context_window: 16384
  temperature: 0.2

advanced:
  enable_semantic_cache: true
  enable_query_expansion: true # Expand "auth" → "authentication, authorize"
```

---

## 11. Advanced Features

### 11.1 Configuration Comparison (P2)

**Feature:** Side-by-side configuration comparison

**Use Case:** "Should I use bge-m3 or MiniLM for my use case?"

```
┌──────────────────────────────────────────────────────┐
│ Compare Configurations                               │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Configuration A      vs    Configuration B         │
│  (Chat Archives)             (Research Papers)      │
│                                                      │
│  Chunk Size:  300          800                      │
│  Model:       bge-m3        bge-large               │
│  Top-K:       6             4                       │
│                                                      │
│  Indexing:    8 min         12 min                  │
│  Quality:     ★★★★☆         ★★★★★                   │
│  Memory:      8GB           10GB                    │
│  Cost:        $0            $0                      │
│                                                      │
│  [Use Config A] [Use Config B] [Merge]              │
└──────────────────────────────────────────────────────┘
```

### 11.2 A/B Testing Mode (P2)

**Feature:** Index same documents with 2 different configurations, compare quality

```typescript
interface ABTest {
  id: string;
  name: string;
  config_a: RAGConfiguration;
  config_b: RAGConfiguration;
  test_queries: string[];
  results: {
    config_a_wins: number;
    config_b_wins: number;
    ties: number;
    quality_metrics: {
      avg_relevance: number;
      avg_latency_ms: number;
      cache_hit_rate: number;
    };
  };
}
```

### 11.3 Configuration Version History (P2)

**Feature:** Track configuration changes over time

```
┌──────────────────────────────────────────────────────┐
│ Configuration History: fb_messenger_archive         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Jan 15, 2026 - 2:30 PM (Current)                  │
│  └─ Changed chunk_size: 500 → 300                  │
│  └─ Enabled semantic cache                          │
│  └─ Performance: +40% cache hit rate                │
│  [Revert to This Version]                           │
│                                                      │
│  Jan 10, 2026 - 10:15 AM                           │
│  └─ Changed model: bge-small → bge-m3              │
│  └─ Enabled MLX backend                             │
│  └─ Performance: +3.7x faster embedding             │
│  [Revert to This Version]                           │
│                                                      │
│  Dec 28, 2025 - Initial Configuration              │
│  [View Diff] [Restore]                              │
└──────────────────────────────────────────────────────┘
```

---

## 12. MVP Scope (Phase 1 - 4 weeks)

### Week 1: Core Infrastructure
- [x] Next.js 14 project setup with TypeScript
- [x] Tailwind CSS + shadcn/ui components
- [x] Zustand stores for state management
- [x] FastAPI backend skeleton
- [x] Authentication (JWT)

### Week 2: Configuration Interface
- [x] Configuration wizard (6 steps)
- [x] Parameter panels (Document, Embedding, LLM)
- [x] Preset selection (3 presets: Chat, Research, Code)
- [x] Form validation with Zod

### Week 3: Indexing & Query
- [x] Document upload (drag & drop)
- [x] Indexing progress monitor (SSE)
- [x] Query interface with streaming responses
- [x] Source attribution display

### Week 4: Polish & Deploy
- [x] Dashboard with metrics
- [x] Index management (list, delete, update)
- [x] Responsive design (mobile, tablet, desktop)
- [x] Testing & deployment

**Deliverables:**
- Functional RAG configuration interface
- 10-15 minute setup time (vs 90 minutes)
- Basic authentication
- Query interface with streaming

---

## 13. Future Enhancements (Phase 2-3)

### Phase 2: Advanced Features (Weeks 5-8)
- Multi-user support with RBAC
- Configuration presets marketplace
- A/B testing framework
- Advanced parameter tuning (reranking, hybrid search)
- Export/import configurations

### Phase 3: Enterprise Features (Weeks 9-12)
- Team collaboration (shared indexes)
- API key management
- Audit logging
- SOC 2 compliance features
- SSO integration (SAML, OAuth)

---

## 14. Success Metrics

### User Experience Metrics

| Metric | Current | Target (MVP) | Target (6 months) |
|--------|---------|--------------|-------------------|
| Setup Time | 90 min | 15 min | 10 min |
| Setup Success Rate | 8% | 60% | 85% |
| User Satisfaction (1-10) | Unknown | 7+ | 8+ |
| Support Requests/Week | Unknown | <5 | <2 |

### Technical Metrics

| Metric | Target |
|--------|--------|
| Page Load (FCP) | <1.5s |
| API Response Time (p95) | <200ms |
| Indexing Job Start | <500ms |
| Query Submission | <100ms |
| Uptime | 99.5%+ |

### Business Metrics (If SaaS)

| Metric | 3 Months | 6 Months | 12 Months |
|--------|----------|----------|-----------|
| Active Users | 50-100 | 200-500 | 1,000-2,000 |
| Conversion Rate | 5% | 10% | 15% |
| MRR | $500 | $2,500 | $10,000 |

---

## 15. Non-Functional Requirements

### 15.1 Scalability

- Support 1-10 concurrent users (MVP)
- Support 10-100 concurrent users (Phase 2)
- Handle indexes up to 1M vectors
- Graceful degradation under load

### 15.2 Reliability

- 99.5% uptime target
- Automatic error recovery (retry with backoff)
- Graceful fallbacks (vLLM → llama.cpp)
- Data persistence (no loss on refresh)

### 15.3 Maintainability

- Component tests (>80% coverage)
- Storybook for component documentation
- TypeScript strict mode (zero `any` types)
- ESLint + Prettier configured
- Git hooks for quality gates

### 15.4 Observability

- OpenTelemetry instrumentation
- Error tracking (Sentry)
- Performance monitoring (Web Vitals)
- User analytics (privacy-respecting, optional)

---

## 16. Technical Constraints

### 16.1 Browser Support

- Chrome/Edge: Last 2 versions
- Firefox: Last 2 versions
- Safari: Last 2 versions (iOS 15+)
- No IE11 support

### 16.2 Device Support

- Desktop: Full experience (all features)
- Tablet: Optimized layout (collapsible panels)
- Mobile: Query-only interface (configuration on desktop)

### 16.3 Dependencies

**Maximum:**
- Bundle size: 500KB initial, 2MB total
- Dependencies: <50 npm packages (avoid bloat)
- Runtime: Node.js 18+, Python 3.11+

---

## 17. Open Questions

### For Product Decision

1. **Multi-tenancy:** Support multiple users with isolated indexes?
   - **Implication:** Requires user database, authentication, isolation
   - **Complexity:** +40% development time
   - **Decision:** Yes for Phase 2, No for MVP

2. **Cloud hosting:** Deploy frontend to Vercel/Netlify or self-hosted only?
   - **Implication:** Self-hosted = more control, cloud = easier deployment
   - **Decision:** Self-hosted for MVP, cloud option in Phase 2

3. **Monetization:** Free/open-source or freemium SaaS?
   - **Implication:** Affects feature gating, authentication approach
   - **Decision:** Open-source core, optional cloud service

### For Technical Implementation

1. **State Persistence:** Where to save configurations?
   - Option A: localStorage (simple, lost on clear)
   - Option B: SQLite database (persistent, queryable)
   - Option C: PostgreSQL (shared across devices)
   - **Decision:** localStorage for MVP, database for Phase 2

2. **Real-time vs Polling:** How to update indexing progress?
   - Option A: Server-Sent Events (one-way, simple)
   - Option B: WebSocket (two-way, more complex)
   - **Decision:** SSE for MVP (simpler, sufficient)

3. **API Framework:** FastAPI vs Flask vs NestJS?
   - **Decision:** FastAPI (async, OpenAPI docs, type hints)

---

## 18. Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Frontend complexity delays MVP | Medium | High | Use shadcn/ui templates, limit scope |
| Backend integration issues | High | High | Wrap existing code (Option A) |
| Performance degradation | Low | Medium | Load testing, optimization sprints |
| Browser compatibility | Low | Low | Polyfills, transpilation |

### Product Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Users still confused by options | Medium | High | Extensive user testing, simplify defaults |
| 15-min setup still too long | Medium | High | One-click Docker Compose, better UX |
| Low adoption (marketing) | High | High | Focus on chat archive niche, Product Hunt launch |

---

## 19. Development Plan

### Sprint 1: Foundation (Week 1)
**Goal:** Working authentication + basic UI

**Tasks:**
- [ ] Next.js 14 project setup with TypeScript
- [ ] Install dependencies: Tailwind, shadcn/ui, Zustand
- [ ] FastAPI backend with authentication
- [ ] Login/logout flow with JWT
- [ ] Basic layout (header, sidebar, main content)

**Deliverable:** User can log in and see empty dashboard

---

### Sprint 2: Configuration Wizard (Week 2)
**Goal:** User can configure RAG pipeline

**Tasks:**
- [ ] Wizard component with 6 steps
- [ ] Document upload (drag & drop + file browser)
- [ ] Preset selection (3 presets)
- [ ] Parameter forms with validation
- [ ] Configuration preview panel

**Deliverable:** User can configure RAG settings end-to-end

---

### Sprint 3: Indexing Integration (Week 3)
**Goal:** User can start indexing and monitor progress

**Tasks:**
- [ ] POST /api/index/start endpoint
- [ ] GET /api/index/status/:id with SSE
- [ ] Progress monitor component
- [ ] Error handling and retry logic
- [ ] Checkpoint/resume support

**Deliverable:** User can index documents with live progress

---

### Sprint 4: Query Interface (Week 4)
**Goal:** User can query indexed documents

**Tasks:**
- [ ] Query input component
- [ ] POST /api/query endpoint
- [ ] Streaming response with SSE
- [ ] Source attribution display
- [ ] Query history

**Deliverable:** Functional end-to-end RAG system

---

## 20. Acceptance Criteria

### Definition of Done (MVP)

A user should be able to:

1. ✅ Create an account and log in (or skip auth for local use)
2. ✅ Select "Chat Archives" preset
3. ✅ Upload a folder of Messenger HTML exports
4. ✅ Click "Start Indexing" and see live progress
5. ✅ Wait 8-12 minutes while indexing completes
6. ✅ Ask "What did Alice say about vacation?"
7. ✅ See streaming response with source attribution
8. ✅ View previous queries in history
9. ✅ Export conversation as Markdown

**Total Time:** 15 minutes (setup) + 10 minutes (indexing) + 30 seconds (first query)

**Success Criteria:**
- ✅ 60%+ users complete setup successfully
- ✅ <5 support questions per week
- ✅ User satisfaction ≥7/10
- ✅ No credential exposure or security issues

---

## 21. Out of Scope (Not in MVP)

❌ Multi-modal support (images, videos)
❌ Collaborative features (shared indexes, comments)
❌ Mobile app (iOS/Android native)
❌ Offline mode (service workers)
❌ Advanced analytics dashboard
❌ Plugin system for custom retrievers
❌ Automated hyperparameter tuning
❌ Model fine-tuning interface
❌ Vector database alternatives (Pinecone, Weaviate)
❌ Integration with external tools (Notion, Obsidian) - Phase 2

---

## Appendix A: Wireframes

### Home Screen

```
┌────────────────────────────────────────────────────────┐
│ [Logo] RAG Pipeline                      [User ▼]     │
├────────────────────────────────────────────────────────┤
│                                                        │
│     🚀 Welcome to Your Private RAG System             │
│                                                        │
│     Search your documents with AI - 100% local        │
│                                                        │
│     [💬 Chat Archives] [📚 Research] [👨‍💻 Code]      │
│                                                        │
│     Or: [📁 Browse Existing Indexes]                  │
│                                                        │
│     ──────────────────────────────────────────────    │
│                                                        │
│     📊 Your Statistics                                │
│     • 2 indexes created                               │
│     • 106,870 chunks indexed                          │
│     • 234 queries this month                          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Configuration Screen

```
┌────────────────────────────────────────────────────────┐
│ Configure RAG Pipeline                     Step 3/6   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Left Panel (40%)         │  Right Panel (60%)        │
│  ───────────────────────  │  ─────────────────────    │
│                           │                           │
│  📄 Document Processing   │  Preview                  │
│                           │  ┌─────────────────────┐  │
│  Chunk Size: 700         │  │ Chunk 1 (698 chars) │  │
│  [━━━━●━━━━] 700 chars   │  │ ┌─────────────────┐ │  │
│                           │  │ │ The quick brown │ │  │
│  Chunk Overlap: 150       │  │ │ fox jumps...    │ │  │
│  [━━●━━━━━━] 150 chars    │  │ └─────────────────┘ │  │
│                           │  │                     │  │
│  ✓ Extract Metadata       │  │ Overlap: ████      │  │
│  ✓ Clean HTML             │  │                     │  │
│                           │  │ Chunk 2 (701 chars) │  │
│  [Back] [Next]            │  └─────────────────────┘  │
│                           │                           │
└────────────────────────────────────────────────────────┘
```

---

## Appendix B: Component Hierarchy

```
App
├── AuthProvider
│   └── LoginPage
│       ├── LoginForm
│       └── OAuthButtons
│
├── AppLayout
│   ├── Header
│   │   ├── Logo
│   │   ├── Navigation
│   │   └── UserMenu
│   │
│   ├── Sidebar (collapsible)
│   │   ├── IndexList
│   │   ├── QuickActions
│   │   └── SystemStatus
│   │
│   └── MainContent
│       ├── Dashboard
│       │   ├── StatsCards
│       │   ├── RecentQueries
│       │   └── QuickStart
│       │
│       ├── ConfigurationWizard
│       │   ├── StepIndicator
│       │   ├── UseCaseSelector
│       │   ├── DocumentUploader
│       │   ├── PresetSelector
│       │   ├── ParameterPanel
│       │   ├── InfrastructureSelector
│       │   └── ReviewSummary
│       │
│       ├── IndexingMonitor
│       │   ├── ProgressBar
│       │   ├── StageIndicator
│       │   ├── LiveMetrics
│       │   └── ErrorHandler
│       │
│       ├── QueryInterface
│       │   ├── QueryInput
│       │   ├── StreamingResponse
│       │   ├── SourceAttribution
│       │   ├── QueryHistory
│       │   └── FilterPanel
│       │
│       └── IndexManager
│           ├── IndexList
│           ├── IndexDetails
│           └── IndexActions
│
└── Providers
    ├── ThemeProvider (dark mode)
    ├── QueryClient (React Query)
    └── ToastProvider (notifications)
```

---

## Appendix C: API Contract

**Full OpenAPI Specification:** Generate with FastAPI automatic docs at `/docs`

**Example Request/Response:**

```typescript
// POST /api/index/start
Request:
{
  "config": {
    "document_processing": {
      "chunk_size": 300,
      "chunk_overlap": 50,
      "extract_metadata": true
    },
    "embedding": {
      "model": "BAAI/bge-m3",
      "backend": "mlx",
      "batch_size": 128
    },
    // ... full configuration
  },
  "documents": {
    "path": "/Users/frytos/data/messenger_export/",
    "file_types": ["html"]
  }
}

Response:
{
  "job_id": "idx_20260115_143022_a4f2b",
  "status": "started",
  "estimated_time_seconds": 480,
  "estimated_chunks": 10247,
  "sse_url": "/api/index/status/idx_20260115_143022_a4f2b"
}

// SSE Stream: /api/index/status/:job_id
event: progress
data: {"stage": "loading", "progress": 0.35, "current": 437, "total": 1247, "eta_seconds": 120}

event: progress
data: {"stage": "chunking", "progress": 0.65, "current": 6660, "total": 10247}

event: progress
data: {"stage": "embedding", "progress": 0.30, "current": 3074, "total": 10247, "speed": 93}

event: complete
data: {"status": "completed", "total_chunks": 10247, "duration_seconds": 462, "index_name": "fb_messenger_20260115"}
```

---

**This requirements document is ready for:**
1. Technical review with engineering team
2. Design mockups and prototypes
3. Sprint planning and estimation
4. Stakeholder approval

**Next Steps:**
1. Review and approve requirements
2. Create design mockups (Figma)
3. Technical architecture design
4. Sprint 1 planning

**Estimated Total Effort:** 4-6 weeks for MVP (1 developer full-time)
