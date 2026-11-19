# PLE-Titan: Elite Physical-Engineering Agent Harness

High-performance agent harness for the **Physical-Engineering’s Last Exam (PLE)** competition ($400,000 USD prize pool).

Combines the multi-phase decomposition of `plan-verify` with architectural patterns from DeepSeek Harness (`repeat-tool-reminder`, `compaction-tool-result-pruner`, `deliverables-guard`) and strict Python-first deterministic calculation directives.

## Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │               PLE-Titan Harness              │
                       │           (Claude Opus 5.5 / Argon)          │
                       └──────────────────────┬───────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
 ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
 │     1. MÓDULO PLAN      │     │    2. MÓDULO WORKER     │     │   3. MÓDULO VERIFIER    │
 │  - Ingestão de CAD/IFC  │ ──► │  - Python-First Math    │ ──► │  - Rubric Self-Grader   │
 │  - Mapeamento de Normas │     │  - DeepSeek Anti-Loop   │     │  - Deliverables Enforcer│
 │  - Roteiro de Cálculo   │     │  - Context Pruner       │     │  - Double-Check Numérico│
 └─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

## Key Modules

| File | Purpose |
|---|---|
| `serve.py` | WebSocket server entrypoint invoking `TitanEngine`. |
| `titan_engine.py` | Core agent loop orchestrating Planner (10%), Worker (65%), and Verifier (25%) phases. |
| `titan_guard.py` | DeepSeek-style `RepeatToolGuard` (anti-loop) and `DeliverablesGuard` (file verification). |
| `titan_compaction.py` | DeepSeek-style tool output pruner and dynamic conversation compactor. |
| `titan_prompts.py` | Domain-specific physical engineering system directives and role prompts. |
| `taskbridge.py` | Official competition WebSocket protocol bridge. |
| `submission.yaml` | Submission declaration with `google-cloud/claude-opus-5-5` and `uv` build instructions. |
| `requirements.txt` | Minimal pinned Python dependencies (`openai`, `websockets`). |
| `artifacts/` | Dev-set execution outputs for competition evaluation. |
