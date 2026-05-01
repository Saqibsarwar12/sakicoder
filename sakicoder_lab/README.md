# sakicoder_lab

This package contains the research and data governance helpers for SakiCoder.

Author attribution for the parent project:
Developed by Saqib Sarwar
Solo developer from Srinagar, J&K, India

It is intended to support a safe MCP-ready workflow for future data collection, validation, experiment logging, and dataset governance. It does not execute remote code, scrape arbitrary content, or bypass license checks.

Main modules:
- `sources.py`: JSONL source metadata handling
- `licenses.py`: license allowlist and warnings
- `quality_filters.py`: content filtering and secret redaction
- `dataset_cards.py`: markdown dataset card generation
- `research_queue.py`: JSONL research task queue
- `experiment_tracker.py`: JSONL experiment logging
- `safety.py`: command safety helpers

See `mcp_plan.md` for the longer workflow and security guidance.
