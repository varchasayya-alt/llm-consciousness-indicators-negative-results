# Research repository: computational pathways toward consciousness-related properties in LLM-based systems

**Current phase:** write-up (2026-10-09). All research lines are closed or stopped; see `report/technical_report.md` and the root `README.md`. Hard constraint throughout: $0 compute (CPU-only laptop).

> The rest of this file was written at Phase 0.5 (2026-10-01) and is kept as a historical orientation. Its "current recommendation" and status statements are superseded by `logs/decisions.md` (D1–D80) and the technical report.

## Start here

- `report/technical_report.md`: **the write-up of the whole program** (results, failures, lessons R1–R12).
- `memo/decision_document_v2.md`: the Phase-0.5 recommendation (P5\* + N1, with P1\* as bridge; P2 deferred), with zero-budget execution plans. Historical.
- `memo/exploratory_research_memo.md`: the main document. §10 is the novelty audit (v0.2); §6–§7 are superseded. It covers problem decomposition, the theory landscape, existing work and gaps, seven candidate programs, a comparison, the preferred direction, and decisive pilots.
- `hypotheses/research_directions.md`: status tracker for programs P1–P7.
- `logs/decisions.md`: why things were decided.

## Layout

```
research/
  memo/                     exploratory memo (versioned in-file)
  literature/
    literature_db.json      SOURCE OF TRUTH for all references (edit this)
    literature_matrix.csv   generated: claim / relevance / limitations / verification per paper
    bibliography.bib        generated BibTeX
    papers/                 PDFs or notes on individual papers (as read)
    theory_notes/           longer theory notes (as needed)
  hypotheses/               research_directions.md, rejected_ideas.md, open_questions.md
  architecture/             specifications/, diagrams/ (empty until a system is designed)
  experiments/
    pilots/pilot_protocols.md   draft preregistrations for Pilots A-D
    baselines/, ablations/
  results/                  raw/ (immutable JSONL), processed/, figures/, statistics/
  paper/                    paper.tex, references.bib, figures/ (later phase)
  logs/                     decisions.md, research_log.md, experiment_log.md
  tools/build_literature.py regenerates matrix + .bib from the JSON DB
  tools/cpu_feasibility_bench.py   CPU throughput benchmark (feasibility only)
  experiments/prototypes/   feasibility-only scripts (no hypotheses tested; burned seeds)
```

## Conventions

- **Claims levels:** 1 = computational result; 2 = theory mapping (always conditional); 3 = phenomenal (never asserted).
- **Verification flags in the literature DB:**
  - `web-2026-10-01`: bibliographic details checked online (content mostly from abstracts).
  - `memory-verify`: entered from memory; check before citing.
  - `partial`: incomplete details.
- **Novelty language:** "we have not yet identified prior work that…" until a full-text check is done.
- **Regenerate the bibliography** after editing the DB:

```bash
python research/tools/build_literature.py
```

## Environment

CPU-only.

- A local virtualenv (`.venv/`, git-ignored) provides CPU PyTorch and transformers.
- The Hugging Face cache lives in `hf_cache/` (git-ignored); run with `HF_HOME=hf_cache`.
- Model weights are never committed.
