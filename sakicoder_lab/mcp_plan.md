# SakiCoder MCP-Ready Research Plan

SakiCoder is a from-scratch PyTorch Transformer project. MCP is planned only for development automation, research, dataset preparation, and testing. It must not become a hosted-model wrapper or an internet-scraping pipeline that blindly trains on unvetted data.

## Useful MCP Servers Later

- GitHub MCP: read repositories, issues, pull requests, workflows, code examples, and project structures.
- Filesystem MCP: read and write files inside the workspace.
- Git MCP: inspect diffs, commits, branches, and code review changes.
- Fetch/Web MCP: read official documentation pages and release notes.
- Playwright MCP: browser-based testing of generated websites and demos.
- Context/documentation MCP: fetch up-to-date library docs and examples.
- Database MCP: store dataset metadata, benchmark results, and experiment logs.

## Security Rules

- Use only trusted MCP servers.
- Allowlist tools before use.
- Block destructive operations by default.
- Never execute unknown remote code.
- Never ingest data without license and source metadata.
- Never train directly from raw scraped content.
- Require dataset validation before training.
- Filter and redact sensitive content before examples enter training sets.
- Keep source provenance with every example and experiment.

## Intended Workflow

1. Create a research queue for desired docs, repos, and examples.
2. Use MCP tools manually or through an agent to collect candidate sources.
3. Save source metadata with license and provenance fields.
4. Validate licenses and source quality.
5. Filter and redact dataset examples.
6. Build a dataset card.
7. Train only on approved filtered data.
8. Log experiment results and checkpoints.
9. Review outputs before expanding to bigger corpora.

## Data Policy

- Prefer official documentation, permissively licensed source code, and first-party examples.
- Reject unknown, private, or unsafe sources.
- Reject content containing secrets, private keys, or suspicious command payloads.
- Keep raw source data separate from filtered training data.
