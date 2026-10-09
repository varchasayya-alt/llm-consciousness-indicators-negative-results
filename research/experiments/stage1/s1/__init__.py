"""Stage-1 package: causal metacognitive monitoring in a controlled synthetic system.

Modules
-------
config        configuration loading and deterministic seed derivation
world         synthetic entity/relation/value world, splits, item sets, sequences
store         first-order store (tiny causal transformer) + training + read-out helpers
matching      read-site displacement metric and caliper matching (frozen algorithm)
interventions store interventions (T-FORGET+sham, T-INTERF, T-NEW, T-FAM, T-ACT, REPLACE)
diagnostics   generic-intervention distinguishability diagnostics (methods development only)
monitors      second-order monitors (IN, OUT, INT-*) and their training pools
controller    ANSWER/LOOKUP controller
metrics       AUROC with ties, d', partial Spearman, TOST, Holm
pipeline      confirmatory per-seed pipeline (guarded: refuses to run unless protocol is frozen)
analysis      confirmatory analysis (reads raw JSONL only)
fake_data     fabricated results for the analysis dry run (no model information)
"""
