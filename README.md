# Octocode — Semantic Code Search Engine & MCP Server

A production-quality, local-first code intelligence engine that provides semantic search, knowledge graphs, hybrid BM25+vector retrieval, multi-source documentation indexing, and a full MCP server for AI assistant integration.

Built on [Muvon/octocode](https://github.com/Muvon/octocode) (Apache 2.0), extended with Azure OpenAI support, hybrid search, external documentation indexing, and branch-aware storage.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Embedding Providers](#embedding-providers)
  - [Search Configuration](#search-configuration)
  - [GraphRAG Configuration](#graphrag-configuration)
- [CLI Reference](#cli-reference)
  - [index](#index)
  - [search](#search)
  - [view](#view)
  - [watch](#watch)
  - [mcp](#mcp)
  - [mcp-proxy](#mcp-proxy)
  - [graphrag](#graphrag)
  - [commit](#commit)
  - [review](#review)
  - [release](#release)
  - [config](#config)
  - [clear](#clear)
  - [format](#format)
  - [logs](#logs)
  - [models](#models)
- [MCP Server](#mcp-server)
  - [Setup with Claude Code](#setup-with-claude-code)
  - [Setup with Cursor](#setup-with-cursor)
  - [MCP Tools Reference](#mcp-tools-reference)
- [Embedding Providers](#embedding-providers-1)
  - [Voyage AI (Default)](#voyage-ai-default)
  - [Azure OpenAI](#azure-openai)
  - [OpenAI](#openai)
  - [Jina AI](#jina-ai)
  - [FastEmbed (Local)](#fastembed-local)
- [Hybrid Search](#hybrid-search)
- [Multi-Source Indexing](#multi-source-indexing)
- [Branch-Aware Indexing](#branch-aware-indexing)
- [GraphRAG Knowledge Graph](#graphrag-knowledge-graph)
- [Supported Languages](#supported-languages)
- [Storage & Data](#storage--data)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Octocode indexes your codebase using tree-sitter AST parsing, generates vector embeddings for semantic understanding, and stores everything in LanceDB for fast retrieval. It combines:

- **Semantic vector search** — find code by what it does, not exact text matches
- **BM25 keyword search** — traditional full-text search via LanceDB FTS indexes
- **Hybrid search with RRF fusion** — combines both signals for 15-30% better recall
- **GraphRAG knowledge graph** — maps relationships between files, functions, and modules
- **Multi-source indexing** — index external documentation alongside your code
- **Real-time file watching** — incremental re-indexing on file changes
- **MCP server** — expose all capabilities to AI assistants via Model Context Protocol

## Architecture

```
                        ┌─────────────────────────────────┐
                        │      MCP Server (stdio/HTTP)     │
                        │                                  │
                        │  semantic_search  view_signatures │
                        │  add_source  remove_source       │
                        │  list_sources  index_source       │
                        │  graphrag (optional)              │
                        │  LSP tools (optional)             │
                        └──────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
    │   Hybrid Search    │   │  Source Indexing   │   │     GraphRAG      │
    │                    │   │                    │   │                    │
    │  Vector (Cosine)   │   │  Code (tree-sitter)│   │  Node search      │
    │  BM25 (FTS)        │   │  Text (line-based) │   │  Get relationships │
    │  RRF Fusion        │   │  Docs (sections)   │   │  Find paths       │
    │  Reranker (opt.)   │   │  URLs (HTML→MD)    │   │  Overview         │
    └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │         Embedding Layer          │
                        │                                  │
                        │  voyage  azure  openai  jina     │
                        │  google  fastembed  huggingface   │
                        │  openrouter  octohub              │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │       LanceDB Storage            │
                        │                                  │
                        │  code_blocks (vector + FTS)      │
                        │  text_blocks (vector + FTS)      │
                        │  document_blocks (vector + FTS)  │
                        │  graphrag_nodes + relationships   │
                        │  metadata (git hash, mtimes)     │
                        └──────────────────────────────────┘
```

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/crisso2292/octocode.git
cd octocode
cargo build --release
# Binary at ./target/release/octocode
```

### With Cargo

```bash
cargo install --path .
```

### Requirements

- Rust 1.82+ (set in `rust-toolchain.toml`)
- At least one embedding API key (Voyage AI recommended — free tier: 200M tokens/month)

---

## Configuration

Octocode uses a TOML configuration file. On first run, the default template is written to `~/.local/share/octocode/config.toml`.

### View/Edit Configuration

```bash
# Show current config
octocode config show

# Set individual values
octocode config set embedding.code_model "azure:text-embedding-3-large"
octocode config set search.hybrid.enabled true

# Reset to defaults
octocode config reset
```

### Full Configuration Reference

```toml
version = 1

[llm]
model = "openrouter:openai/gpt-4o-mini"    # For commit, review, release features
timeout = 120
temperature = 0.7
max_tokens = 4000

[index]
chunk_size = 2000                           # Max characters per code chunk
chunk_overlap = 100                         # Overlap between adjacent chunks
embeddings_batch_size = 16                  # Files per embedding batch
embeddings_max_tokens_per_batch = 100000    # Token limit per API call
flush_frequency = 2                         # Flush to disk every N batches
require_git = true                          # Require git repo for indexing

[search]
max_results = 20                            # Default result count
similarity_threshold = 0.65                 # Min similarity (0.0-1.0)
output_format = "markdown"                  # markdown | json | text
max_files = 10
context_lines = 3
search_block_max_characters = 400           # Max chars per result block

[search.reranker]
enabled = false                             # Enable Voyage/Cohere/Jina reranking
model = "voyage:rerank-2.5"                 # Reranker model
top_k_candidates = 50                       # Candidates before reranking
final_top_k = 10                            # Results after reranking

[search.hybrid]
enabled = true                              # Hybrid BM25 + vector (recommended)
default_vector_weight = 0.7                 # Vector signal weight
default_keyword_weight = 0.3                # Keyword signal weight
keyword_path_weight = 2.0                   # Boost for path matches
keyword_content_weight = 1.0                # Boost for content matches
keyword_symbols_weight = 2.5                # Boost for symbol matches
keyword_title_weight = 3.0                  # Boost for title matches (docs)

[embedding]
code_model = "voyage:voyage-code-3"         # Best code embedding model
text_model = "voyage:voyage-3.5-lite"       # Text/doc embedding model

[graphrag]
enabled = false                             # Enable knowledge graph
use_llm = false                             # Use LLM for relationship discovery
```

### Embedding Providers

| Provider | Format | Env Variable | Models |
|----------|--------|-------------|--------|
| **Voyage AI** | `voyage:model` | `VOYAGE_API_KEY` | voyage-code-3, voyage-3.5-lite |
| **Azure OpenAI** | `azure:model` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | text-embedding-3-large (3072d), text-embedding-3-small (1536d) |
| **OpenAI** | `openai:model` | `OPENAI_API_KEY` | text-embedding-3-large, text-embedding-3-small |
| **Jina AI** | `jina:model` | `JINA_API_KEY` | jina-embeddings-v2-base-code |
| **Google** | `google:model` | `GOOGLE_API_KEY` | text-embedding-004 |
| **FastEmbed** | `fastembed:model` | None (local) | jinaai/jina-embeddings-v2-base-code |
| **HuggingFace** | `huggingface:model` | None (local) | Various GGUF models |
| **OpenRouter** | `openrouter:model` | `OPENROUTER_API_KEY` | Dynamic model discovery |

### Search Configuration

Hybrid search combines two complementary signals:

- **Vector search (cosine similarity)**: Understands semantic meaning — "authentication middleware" finds auth code even if those exact words aren't in the code
- **BM25 keyword search (FTS)**: Exact term matching — finds `authenticate()` when you search for "authenticate"
- **RRF fusion**: Reciprocal Rank Fusion merges both result sets, giving 15-30% better recall than either alone

### GraphRAG Configuration

When enabled (`graphrag.enabled = true`), octocode builds a knowledge graph mapping relationships between files: imports, inheritance, function calls, pattern usage. Enable `use_llm = true` for AI-powered relationship discovery (requires LLM API key).

---

## CLI Reference

### index

Index the current directory's codebase. Parses source files with tree-sitter, generates embeddings, and stores in LanceDB.

```bash
octocode index                    # Index current directory
octocode index --quiet            # Suppress progress output
```

Uses differential indexing: only re-processes files that changed since the last index (based on SHA-256 content hashes and file modification times).

### search

Search the indexed codebase with natural language queries.

```bash
# Single query
octocode search "HTTP request handling with error recovery"

# Multi-query (broader coverage, results deduplicated)
octocode search "authentication" "JWT token validation" "session management"

# Filter by content type
octocode search "database connection pooling" --mode code
octocode search "API documentation" --mode docs

# Filter by language
octocode search "error handling patterns" --language rust

# Control output
octocode search "config parsing" --max-results 5 --detail full
octocode search "routing" --format json
```

**Options:**
- `--mode <code|text|docs|all>` — Filter by content type (default: all)
- `--detail <signatures|partial|full>` — Result verbosity (default: partial)
- `--max-results <N>` — Maximum results (default: 20)
- `--threshold <0.0-1.0>` — Similarity cutoff (default: 0.65)
- `--language <lang>` — Filter by programming language
- `--format <markdown|json|text>` — Output format

### view

Extract function signatures, class definitions, and declarations from files.

```bash
octocode view src/main.rs                    # Single file
octocode view "src/**/*.rs"                  # Glob pattern
octocode view src/config.rs src/store/mod.rs # Multiple files
```

### watch

Watch for file changes and automatically re-index.

```bash
octocode watch                    # Watch current directory
octocode watch --debounce 3000   # Custom debounce (ms)
```

### mcp

Start the MCP (Model Context Protocol) server for AI assistant integration.

```bash
# Standard stdio mode (for Claude Code, Cursor, etc.)
octocode mcp --path /path/to/project

# HTTP mode (for custom clients)
octocode mcp --path /path/to/project --bind 0.0.0.0:12345

# With LSP integration
octocode mcp --path /path/to/project --with-lsp "rust-analyzer"

# Without git requirement
octocode mcp --path /path/to/project --no-git

# Debug mode (verbose logging)
octocode mcp --path /path/to/project --debug
```

### mcp-proxy

Start a multi-repository MCP proxy server. Scans a root directory for git repositories and serves them all via a single HTTP endpoint.

```bash
octocode mcp-proxy --root ~/projects --bind 0.0.0.0:9090
```

### graphrag

Query the code knowledge graph.

```bash
# Search nodes semantically
octocode graphrag search "authentication flow"

# Get node details
octocode graphrag get-node src/auth/middleware.rs

# Get relationships for a node
octocode graphrag get-relationships src/config.rs

# Find paths between nodes
octocode graphrag find-path src/main.rs src/store/mod.rs --max-depth 4

# Get graph overview
octocode graphrag overview

# JSON output
octocode graphrag search "error handling" --format json
```

### commit

Generate AI-powered git commit messages from staged changes.

```bash
octocode commit                  # Commit staged changes
octocode commit --all            # Stage all and commit
```

### review

Review staged changes for best practices and potential issues.

```bash
octocode review                  # Review staged changes
```

### release

Create a release with AI-powered version calculation and changelog generation.

```bash
octocode release                 # Auto-calculate version bump
octocode release --bump minor    # Force minor version bump
```

### config

Manage configuration.

```bash
octocode config show             # Show current config
octocode config set key value    # Set a config value
octocode config reset            # Reset to defaults
```

### clear

Clear database tables.

```bash
octocode clear                   # Clear all tables
octocode clear --table code      # Clear only code blocks
octocode clear --table docs      # Clear only document blocks
octocode clear --table text      # Clear only text blocks
```

### format

Format code according to `.editorconfig` rules.

```bash
octocode format                  # Format all files
octocode format src/main.rs      # Format specific file
```

### logs

View MCP server logs.

```bash
octocode logs                    # Show recent logs
octocode logs --follow           # Follow log output
```

### models

Model management and discovery.

```bash
octocode models list             # List available models
octocode models info voyage      # Show provider details
```

---

## MCP Server

The MCP server exposes octocode's capabilities to AI assistants via the [Model Context Protocol](https://modelcontextprotocol.io/). It runs as a subprocess communicating over stdin/stdout (JSON-RPC).

### Setup with Claude Code

Add to your Claude Code MCP settings (`~/.claude/settings.json` or project-level):

```json
{
  "mcpServers": {
    "octocode": {
      "command": "/path/to/octocode",
      "args": ["mcp", "--path", "/path/to/your/project"],
      "env": {
        "VOYAGE_API_KEY": "your-voyage-key"
      }
    }
  }
}
```

### Setup with Cursor

Add to Cursor's MCP configuration:

```json
{
  "mcpServers": {
    "octocode": {
      "command": "/path/to/octocode",
      "args": ["mcp", "--path", "/path/to/your/project"]
    }
  }
}
```

### MCP Tools Reference

#### `semantic_search`

Search the codebase by meaning. The primary search tool.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string or string[] | *required* | What to search for. Array preferred for broader coverage. |
| `mode` | string | `"all"` | `code`, `text`, `docs`, or `all` |
| `detail_level` | string | `"partial"` | `signatures`, `partial`, or `full` |
| `max_results` | integer | `3` | 1-20 |
| `threshold` | number | `0.65` | Similarity cutoff (0.0-1.0) |
| `language` | string | — | Filter by language (e.g., `"rust"`, `"python"`) |
| `search_strategy` | string | `"hybrid"` | `hybrid` (vector+BM25), `vector` (semantic only), `keyword` (BM25 only) |

**Example:**
```json
{
  "query": ["authentication middleware", "JWT token validation"],
  "mode": "code",
  "search_strategy": "hybrid",
  "max_results": 5,
  "language": "rust"
}
```

#### `view_signatures`

Extract function signatures and declarations from files without implementation bodies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `files` | string[] | *required* | File paths or glob patterns |

**Example:**
```json
{
  "files": ["src/config.rs", "src/store/**/*.rs"]
}
```

#### `add_source`

Register an external documentation URL for indexing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | *required* | Unique identifier (e.g., `"tokio-docs"`) |
| `url` | string | *required* | URL to fetch and index |
| `type` | string | `"url"` | `url` (single page) or `sitemap` (multiple pages) |

#### `remove_source`

Remove an external documentation source and its indexed content.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | *required* | Name of the source to remove |

#### `list_sources`

List all configured external documentation sources with their indexing status. No parameters.

#### `index_source`

Fetch, convert, chunk, embed, and store content from a registered source.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | *required* | Name of the source to index |

**Workflow:**
```
add_source("tokio-docs", "https://docs.rs/tokio/latest/tokio/")
  → index_source("tokio-docs")
  → semantic_search("async runtime executor", mode="docs")
```

#### `graphrag` (when enabled)

Knowledge graph operations for architectural queries.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | string | *required* | `search`, `get-node`, `get-relationships`, `find-path`, `overview` |
| `query` | string | — | Search query (for `search` operation) |
| `node_id` | string | — | Node ID (for `get-node`, `get-relationships`) |
| `source_id` | string | — | Source node (for `find-path`) |
| `target_id` | string | — | Target node (for `find-path`) |
| `max_depth` | integer | `3` | Max path depth (for `find-path`) |
| `format` | string | `"text"` | `text`, `json`, or `markdown` |

---

## Embedding Providers

### Voyage AI (Default)

Best quality for code search. Free tier: 200M tokens/month.

```bash
export VOYAGE_API_KEY="pa-..."
```

```toml
[embedding]
code_model = "voyage:voyage-code-3"      # 1024 dimensions, best for code
text_model = "voyage:voyage-3.5-lite"    # 1024 dimensions, fast for text
```

### Azure OpenAI

For organizations with Azure OpenAI deployments. Deployment name must match model name.

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

```toml
[embedding]
code_model = "azure:text-embedding-3-large"   # 3072 dimensions
text_model = "azure:text-embedding-3-large"
```

**Supported models:**

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `text-embedding-3-large` | 3072 | Highest quality |
| `text-embedding-3-small` | 1536 | Good balance of quality/cost |
| `text-embedding-ada-002` | 1536 | Legacy model |

### OpenAI

Direct OpenAI API access.

```bash
export OPENAI_API_KEY="sk-..."
```

```toml
[embedding]
code_model = "openai:text-embedding-3-large"
text_model = "openai:text-embedding-3-small"
```

### Jina AI

Good for code, with native input type support.

```bash
export JINA_API_KEY="jina_..."
```

```toml
[embedding]
code_model = "jina:jina-embeddings-v2-base-code"
text_model = "jina:jina-embeddings-v2-base-code"
```

### FastEmbed (Local)

No API key needed. Runs entirely offline using ONNX models.

```toml
[embedding]
code_model = "fastembed:jinaai/jina-embeddings-v2-base-code"
text_model = "fastembed:sentence-transformers/all-MiniLM-L6-v2-quantized"
```

Requires the `fastembed` feature flag (enabled by default).

---

## Hybrid Search

Hybrid search is enabled by default and combines two complementary retrieval strategies:

### How It Works

1. **Vector search**: The query is embedded and compared against stored embeddings using cosine similarity. This understands semantic meaning — "error handling" finds `try/catch` blocks even without exact word matches.

2. **BM25 keyword search**: The raw query string is matched against a full-text search index on the `content` column. This finds exact terms — searching "authenticate" finds functions named `authenticate`.

3. **Reciprocal Rank Fusion (RRF)**: Results from both signals are merged. Items that rank highly in both lists get the strongest boost. The formula: `score = 1/(k + rank_vector) + 1/(k + rank_bm25)` with k=60.

### Configuration

```toml
[search.hybrid]
enabled = true                  # Toggle hybrid search
default_vector_weight = 0.7     # Relative weight of vector signal
default_keyword_weight = 0.3    # Relative weight of keyword signal
```

### Per-Query Control via MCP

The `search_strategy` parameter on `semantic_search` overrides the config per-query:

- `"hybrid"` — Both signals with RRF fusion (default, recommended)
- `"vector"` — Semantic similarity only (good for natural language questions)
- `"keyword"` — BM25 text matching only (good for exact symbol lookup)

### FTS Index Management

LanceDB FTS indexes are created lazily — they're built on first use when hybrid search is enabled. Subsequent queries reuse the existing index. Indexes are automatically updated when new data is stored.

---

## Multi-Source Indexing

Index external documentation alongside your code so AI assistants can reference both.

### Via MCP Tools

```
1. add_source(name="react-docs", url="https://react.dev/reference/react")
2. index_source(name="react-docs")
3. semantic_search("component lifecycle hooks", mode="docs")
```

### How It Works

1. **Fetch**: HTTP GET the URL with proper headers
2. **Convert**: HTML is converted to clean markdown (scripts, nav, footer stripped)
3. **Chunk**: Content is split into sections by heading structure (h1/h2/h3)
4. **Embed**: Each section is embedded using the configured text model
5. **Store**: Sections are stored as `DocumentBlock` entries with `source://{name}/` path prefix

### Source Management

Sources are persisted per-project in `sources.toml` within the project's storage directory. They survive re-indexing and can be independently updated.

```bash
# Sources are stored at:
# ~/.local/share/octocode/{project_hash}/sources.toml
```

---

## Branch-Aware Indexing

Inspired by Augment Code's per-developer branch views. When enabled, each git branch gets its own independent search index.

### Enable

```bash
export OCTOCODE_BRANCH_AWARE=1
```

### How It Works

When `OCTOCODE_BRANCH_AWARE=1`:
- Storage path changes from `storage/` to `branches/{branch_name}/`
- Each branch has its own LanceDB database with independent embeddings
- Switching branches automatically switches the search index
- Branch names with `/` (e.g., `feature/auth`) are sanitized to `feature__auth`

### When to Use

- **Feature branches** with significant code changes that affect search relevance
- **Experimentation** where you want isolated indexes
- **Multi-developer** setups where each dev's branch has different code

### When NOT to Use

- Small repos where branch differences are minimal
- Storage-constrained environments (each branch duplicates the full index)

---

## GraphRAG Knowledge Graph

When enabled, octocode builds a knowledge graph mapping architectural relationships between files.

### Enable

```toml
[graphrag]
enabled = true
use_llm = false    # true for AI-powered relationship discovery (requires LLM API key)
```

### Relationship Types

Without LLM (`use_llm = false`): Relationships are discovered through static analysis — imports, exports, function calls, class inheritance, module structure.

With LLM (`use_llm = true`): AI analyzes code to discover architectural patterns — factory patterns, observer patterns, strategy patterns, dependency injection, adapter patterns.

### Query Examples

```bash
# What does this file depend on?
octocode graphrag get-relationships src/auth/middleware.rs

# How are these two files connected?
octocode graphrag find-path src/main.rs src/store/mod.rs

# What files are related to authentication?
octocode graphrag search "authentication and authorization flow"

# Graph statistics
octocode graphrag overview
```

---

## Supported Languages

| Language | Extensions | AST Parsing | Features |
|----------|-----------|-------------|----------|
| **Rust** | `.rs` | tree-sitter-rust | Functions, structs, traits, impls, pub/use, modules |
| **Python** | `.py` | tree-sitter-python | Functions, classes, imports, decorators, docstrings |
| **JavaScript** | `.js`, `.jsx` | tree-sitter-javascript | Functions, classes, ES6 imports/exports, arrow functions |
| **TypeScript** | `.ts`, `.tsx` | tree-sitter-typescript | Types, interfaces, generics, decorators, namespaces |
| **Go** | `.go` | tree-sitter-go | Functions, structs, interfaces, packages, imports |
| **PHP** | `.php` | tree-sitter-php | Classes, functions, namespaces, traits, interfaces |
| **C++** | `.cpp`, `.hpp`, `.h` | tree-sitter-cpp | Classes, functions, templates, includes, namespaces |
| **Ruby** | `.rb` | tree-sitter-ruby | Classes, modules, methods, blocks, mixins |
| **Java** | `.java` | tree-sitter-java | Classes, interfaces, methods, annotations, packages |
| **Lua** | `.lua` | tree-sitter-lua | Functions, tables, modules |
| **Bash** | `.sh`, `.bash` | tree-sitter-bash | Functions, variables, aliases |
| **CSS** | `.css` | tree-sitter-css | Selectors, properties, media queries |
| **JSON** | `.json` | tree-sitter-json | Structure analysis, key extraction |
| **Svelte** | `.svelte` | tree-sitter-svelte | Components, scripts, styles |
| **Markdown** | `.md` | Header parser | Sections by heading hierarchy |

---

## Storage & Data

### Storage Layout

```
~/.local/share/octocode/
├── config.toml                          # Global configuration
├── fastembed/                           # Local model cache (if using fastembed)
├── sentencetransformer/                 # HuggingFace model cache
└── {project_hash}/                      # Per-project storage
    ├── storage/                         # LanceDB database (default)
    │   ├── code_blocks.lance/           # Code embeddings + content
    │   ├── text_blocks.lance/           # Text embeddings + content
    │   ├── document_blocks.lance/       # Doc embeddings + content
    │   ├── graphrag_nodes.lance/        # Knowledge graph nodes
    │   └── graphrag_relationships.lance/# Knowledge graph edges
    ├── branches/                        # Branch-aware databases (opt-in)
    │   ├── main/                        # main branch index
    │   └── feature__auth/               # feature/auth branch index
    └── sources.toml                     # External source configuration
```

### Project Identification

Projects are identified by a SHA-256 hash of:
1. The git remote URL (if available), ensuring the same repo maps to the same index regardless of local path
2. The absolute filesystem path (fallback for non-git directories)

### Data Safety

- Indexing uses write-ahead logging via LanceDB
- `flush_frequency` controls how often data is written to disk
- File locking prevents concurrent indexing corruption
- Differential indexing skips unchanged files (SHA-256 content hashes + mtime checks)

---

## Performance

### Indexing Speed

- **Differential indexing**: Only processes changed files. First index of a large repo may take minutes; subsequent indexes take seconds.
- **Batch embedding**: Files are batched (default: 16 per batch) to minimize API round-trips.
- **Token-aware batching**: Batches respect both count and token limits to avoid API errors.

### Search Speed

- **LanceDB vector index**: IVF_HNSW_SQ (Inverted File + Hierarchical Navigable Small World + Scalar Quantization) for sub-millisecond ANN search.
- **Table caching**: Frequently-accessed tables are cached in memory to avoid repeated open overhead.
- **Query optimization**: Index parameters (nprobes, refine factor) auto-tuned based on dataset size.
- **Hybrid search**: Vector and FTS queries run in parallel via LanceDB native execution.

### Memory Usage

- Embeddings: ~20 bytes per line of code (for 1024-dimension models)
- LanceDB uses memory-mapped I/O — only accessed pages are loaded
- No in-memory index for small datasets (< 256 rows); IVF_PQ for larger datasets

---

## Troubleshooting

### "Model format must be 'provider:model'"

Your config has an invalid embedding model string. It must be in `provider:model` format:
```bash
octocode config set embedding.code_model "voyage:voyage-code-3"
```

### "VOYAGE_API_KEY environment variable not set"

Set your API key:
```bash
export VOYAGE_API_KEY="pa-your-key-here"
```

### "GraphRAG config must be loaded from template file"

You're seeing a panic from `Config::default()`. This means the config template wasn't loaded properly. Delete the system config and let it regenerate:
```bash
rm ~/.local/share/octocode/config.toml
octocode config show
```

### "Schema mismatch detected"

This happens when you change embedding models (different dimension). Octocode automatically drops and recreates the affected tables. Re-index:
```bash
octocode clear
octocode index
```

### MCP server not responding

Check logs:
```bash
octocode logs
```

Ensure the path is correct and the project has been indexed:
```bash
cd /path/to/project
octocode index
octocode mcp --path /path/to/project --debug
```

### Hybrid search returns fewer results than expected

FTS indexes are created lazily. If you enabled hybrid search after indexing, the FTS index doesn't exist yet. Re-index or run a search to trigger lazy creation:
```bash
octocode clear
octocode index
```

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

Based on [Muvon/octocode](https://github.com/Muvon/octocode) by Muvon Un Limited (Hong Kong).
