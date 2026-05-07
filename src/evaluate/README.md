# CAP-Eval

Evaluation toolkit for agentic search systems, featuring an **Agent-as-a-Judge** methodology for comprehensive, rigorous, and reliable automated assessment on **long-horizon** and complex tasks involving **complex and real-time information synthesis**.

## Overview

CAP-Eval evaluates the ability of AI agents to autonomously browse the web, gather information, and produce long-form answers with proper source attribution. Its core innovation is defining a **Verification Tree** (tree-structured rubric) for each task, where an LLM judge agent automatically assesses answer correctness and source reliability.

### Project Structure

```
run_eval.py                  # Evaluation entry point (CLI)
generate_eval.py             # LLM-based automatic evaluation script generation
run_eval_batch.sh            # Batch evaluation script generation (shell wrapper)
count.py                     # Evaluation result statistics (partial completion, success rate, etc.)

cap_eval/                    # Core Python package
├── evaluator.py             # High-level Evaluator class (extraction + verification orchestration)
├── eval_toolkit.py          # Extractor & Verifier classes (LLM-powered structured extraction/verification)
├── verification_tree.py     # VerificationNode tree structure and aggregation strategies
├── eval_runner.py           # Async task/answer evaluation orchestrator
├── api_tools/               # External API integrations (arXiv, Google Maps, PDF)
├── llm_client/              # LLM provider abstraction (OpenAI, Azure OpenAI, Bedrock)
├── utils/                   # Shared utilities (caching, logging, browser, path config)
└── prompts/                 # Prompt templates for LLM extraction

cache_manager_web/           # Browser-based cache management and repair tool
├── backend/                 # FastAPI backend service
├── frontend/                # Web frontend interface
├── extension/               # Chrome extension (automatic webpage recapture)
└── run.py                   # Launch entry point

answers/                     # Agent answer files (Markdown), organized by agent/task
eval_scripts/                # Per-task evaluation scripts
cache/                       # Webpage cache (text + screenshots)
eval_results/                # Evaluation output (JSON results + logs)
```

## Environment Setup

### Option 1: Using uv (Recommended)

```bash
uv sync
source .venv/bin/activate
patchright install
```

### Option 2: Using conda + pip

```bash
conda create -n cap-eval python=3.11
conda activate cap-eval
pip install -e .
patchright install
```

> **Note**: `patchright install` installs the browser engine (Chromium), which is required for browser automation to retrieve webpage content and screenshots during evaluation.

## Prepare Your Data

Organize your agent's answer files in the following directory structure:

```
answers/
└── <agent_name>/
    └── <task_id>/
        ├── answer_1.md
        ├── answer_2.md
        └── ...
```

Each answer file should be in Markdown format and may contain URL citations.

## Set Up API Keys

```bash
# OpenAI (required)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

# Azure OpenAI (optional)
export AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
export AZURE_OPENAI_ENDPOINT_URL="YOUR_AZURE_OPENAI_ENDPOINT_URL"
export AZURE_OPENAI_API_VERSION="2025-03-01-preview"

# Google Maps API (optional, only needed for certain tasks)
export GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
```

## Run Evaluation

```bash
# Evaluate all tasks for a specific agent
python run_eval.py --agent_name <agent_name>

# Evaluate a specific task
python run_eval.py --agent_name <agent_name> --task_id <task_id>
```

### Full Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agent_name` | (required) | Agent name |
| `--task_id` | None (evaluates all tasks) | Specific task ID to evaluate |
| `--answer_folder` | `answers/` | Directory containing answer files |
| `--eval_scripts_root` | `eval_scripts/` | Root directory for evaluation scripts |
| `--eval_results_root` | `eval_results/` | Root directory for evaluation result output |
| `--cache_root` | `cache/` | Root directory for webpage cache |
| `--eval_version` | `2026_03_13` | Evaluation script version |
| `--llm_provider` | `openai` | LLM provider (`openai` or `azure_openai`) |
| `--max_concurrent_tasks` | `3` | Maximum number of concurrent task evaluations |
| `--max_concurrent_answers` | `3` | Maximum number of concurrent answer evaluations per task |
| `--max_webpage_retrieval` | `10` | Maximum number of concurrent webpage retrievals (Playwright) |
| `--max_llm_requests` | `30` | Maximum number of concurrent LLM API requests |
| `--dump_cache` | `True` | Persist cache to disk after evaluation |
| `--overwrite` | `False` | Overwrite existing evaluation results |
| `--self_debug` | `False` | Debug mode — appends `_debug` suffix to logs and result files |

## Generate Evaluation Scripts

Use `generate_eval.py` to automatically generate evaluation scripts for new tasks via LLM:

```bash
# Generate an evaluation script for a single task
python generate_eval.py --json_path <task_json_path>

# With optional parameters
python generate_eval.py \
    --json_path <task_json_path> \
    --py_path <reference_template.py> \
    --output_dir <output_directory>
```

To batch-generate evaluation scripts for all tasks:

```bash
bash run_eval_batch.sh
```

This script iterates over all `task-*` directories under `answers/`, locates the corresponding JSON task files, and invokes `generate_eval.py` in parallel (default concurrency: 7).

## Cache Manager

`cache_manager_web/` provides a browser-based visual cache management interface for inspecting and repairing webpage caches used during evaluation. Paired with a Chrome extension, it can **automatically batch-recapture** problematic pages (CAPTCHA walls, access denied, blank pages, etc.).

```bash
# Launch by agent name (auto-locates cache/<agent_name>)
uv run python3 cache_manager_web/run.py <agent_name>

# Launch by full path
uv run python3 cache_manager_web/run.py /path/to/cache/folder
```

Key features:
- **Three-panel layout**: Task list → URL list → Content preview (toggle between screenshot/text/answer)
- **Issue detection**: Automatically flags problematic cached pages (red = definite issue, yellow = possible issue)
- **Batch repair**: One-click queue for all flagged pages; Chrome extension auto-opens and recaptures
- **CAPTCHA handling**: Automatically pauses on CAPTCHA detection, waits for manual resolution, then continues
- **Keyboard shortcuts**: `j`/`k` to navigate, `n`/`N` to jump across issues, `r` to confirm fix, etc.

See [cache_manager_web/README.md](cache_manager_web/README.md) for details.

## Evaluation Results

Evaluation results are stored in `eval_results/<agent_name>/`. Use `count.py` to compute aggregate statistics:

```bash
python count.py
```

Output metrics include:
- **Partial Completion**: Average root node score across all tasks
- **Success Rate**: Proportion of tasks with a root node score of 1.0
- **Complex A**: Average score of Action nodes
- **Complex P**: Average score of Perception nodes

## Core Concepts

### Evaluation Pipeline

1. **Agent produces answers** — Markdown files with URL citations, stored in `answers/<agent>/<task>/answer_*.md`
2. **Evaluation scripts define judging logic** — Each task has a Python script defining an `async def evaluate_answer(...)` function that builds a Verification Tree
3. **LLM judge agent executes evaluation** — Extracts structured information from answers and verifies each claim against source webpages

### Verification Tree

- A tree-structured rubric where each node is a `VerificationNode`
- Leaf nodes receive binary pass/fail scores via LLM verification
- Two aggregation strategies:
  - **PARALLEL**: Weighted average (evaluates all child nodes in parallel)
  - **SEQUENTIAL**: Short-circuits on failure (evaluates in order; skips remaining nodes if a predecessor fails)
- **Critical nodes**: If any critical child node fails, the parent node scores 0.0 (acts as a gate)

### Evaluator API (Used in Evaluation Scripts)

```python
evaluator = Evaluator()
root = evaluator.initialize(task_id=..., agent_name=..., answer_name=...)

# Extract structured information from the answer
info = await evaluator.extract(prompt, TemplateClass)

# Verify a claim (optionally specifying source URLs)
await evaluator.verify(claim="...", node=leaf_node, sources="https://...")

# Get the final result
return evaluator.get_summary()
```

### Concurrency Model

- Fully asynchronous using `asyncio`
- Semaphores control concurrency at multiple levels: tasks, answers, webpage retrieval, LLM requests
- `DualSemaphore` wraps both webpage retrieval and LLM request concurrency controls

## Key Dependencies

| Dependency | Purpose |
|------------|---------|
| `patchright` | Stealth browser automation for webpage capture (Playwright fork) |
| `openai` | LLM API calls (OpenAI / Azure OpenAI) |
| `pydantic` | Structured output parsing from LLM responses |
| `PyMuPDF (fitz)` | PDF parsing and rendering |
| `html2text` | HTML to Markdown conversion |
| `beautifulsoup4` | HTML parsing |
| `backoff` | Exponential backoff retry for LLM calls |
| `arxiv` | arXiv paper search API |
| `googlemaps` | Google Maps API integration |

## License

See [LICENSE](LICENSE).
