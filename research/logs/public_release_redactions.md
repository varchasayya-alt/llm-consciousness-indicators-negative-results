# Public-release redactions

Made in the commit that added `research/report/technical_report.md` (write-up phase, 2026-10-09).

**What was removed.** The local absolute path of the repository checkout, which contained the author's OS user name. It was the only sensitive string found: no tokens, keys, passwords, e-mail addresses or host names occur in any tracked file.

**Rules.**
- *Records* (logs, recorded commands): the path is replaced by the literal `<REPO_ROOT>`. Nothing else changed; replacing `<REPO_ROOT>` with the original path restores the original bytes exactly (checked per file before writing).
- *Launchers* still used for reproduction: the hard-coded `set "ROOT=..."` line is replaced by `for %%I in ("%~dp0..\..\..") do set "ROOT=%%~fI"` (the repository root derived from the script's location). No other line changed.
- No result value, threshold, material or code path used by any analysis was changed.

**Consequence for integrity pins.**
- 12 redacted records sit under `research/results/raw/c15a/`, a tree whose git hash is pinned by the C15-R2 guards (`verdict2.json`, tree `54a3f123…`). After this commit that pin no longer matches HEAD, so `run_r2.py` refuses to run at HEAD by design.
- `_r2_eng_chain.cmd` and `_r2_select_chain.cmd` are hashed in `research/results/raw/c15r2/prerun/prerun_record.json`; those hashes refer to the pre-redaction files.
- To verify or re-run an experiment, check out its result commit (see the root `README.md`), where every pin matches. The original bytes are in the parent commit; their blob SHAs are listed below.

| File | Occurrences | Change | Original blob | Redacted blob |
|---|---|---|---|---|
| `research/experiments/c15r2/_r2_eng_chain.cmd` | 1 | launcher: ROOT now derived from the script location | `53ff312d580d` | `f11a794bc9ef` |
| `research/experiments/c15r2/_r2_run_phase.cmd` | 1 | launcher: ROOT now derived from the script location | `472a7ef8399d` | `a047a57a44e7` |
| `research/experiments/c15r2/_r2_select_chain.cmd` | 1 | launcher: ROOT now derived from the script location | `7923035c2461` | `c13db8859bd9` |
| `research/experiments/c16/_c16_chain.cmd` | 1 | launcher: ROOT now derived from the script location | `4d97d1ef7e9d` | `d93ee6387ec9` |
| `research/memo/c15r2_power/r2_null_calibration.log` | 1 | record: path → `<REPO_ROOT>` | `dc3b804a6718` | `d631bf36acc3` |
| `research/results/raw/c15a/download_astage.log` | 6 | record: path → `<REPO_ROOT>` | `3a30ec577002` | `99e83f8fdd39` |
| `research/results/raw/c15a/engineering/smoke_qwen3-1.7b.log` | 1 | record: path → `<REPO_ROOT>` | `2a8203a9e23c` | `0087ecef6915` |
| `research/results/raw/c15a/engineering/smoke_qwen3-4b.log` | 1 | record: path → `<REPO_ROOT>` | `634cb05e33d9` | `22b204171e8c` |
| `research/results/raw/c15a/engineering/smoke_qwen3.5-2b.log` | 1 | record: path → `<REPO_ROOT>` | `1231489ee447` | `e9aaaf1fcae9` |
| `research/results/raw/c15a/engineering/validate_qwen3-1.7b.log` | 1 | record: path → `<REPO_ROOT>` | `eff3b0f78113` | `11c5025b8f89` |
| `research/results/raw/c15a/engineering/validate_qwen3-4b.log` | 1 | record: path → `<REPO_ROOT>` | `57a6d2baafbb` | `5033735b98aa` |
| `research/results/raw/c15a/engineering/validate_qwen3.5-2b.log` | 1 | record: path → `<REPO_ROOT>` | `7731544722c6` | `ecf4409164fa` |
| `research/results/raw/c15a/logs/choose.log` | 13 | record: path → `<REPO_ROOT>` | `03f91618ae12` | `6525b6dd1c67` |
| `research/results/raw/c15a/logs/rerun_qwen3-4b.cmd` | 1 | record: path → `<REPO_ROOT>` | `c52269b46cd4` | `e648fb433630` |
| `research/results/raw/c15a/logs/select_qwen3-1.7b.log` | 1 | record: path → `<REPO_ROOT>` | `73a745387fd4` | `e25244c76ff4` |
| `research/results/raw/c15a/logs/select_qwen3-4b.log` | 1 | record: path → `<REPO_ROOT>` | `e76bbbbd4255` | `a979db257b82` |
| `research/results/raw/c15a/logs/select_qwen3.5-2b.log` | 1 | record: path → `<REPO_ROOT>` | `0bf82fba43a8` | `883aef37f874` |
| `research/results/raw/c15r2/logs/classify2.log` | 2 | record: path → `<REPO_ROOT>` | `601aedeb54fa` | `31d23d4be527` |
| `research/results/raw/c15r2/logs/phase_SMOKE2_qwen3.5-2b_.log` | 2 | record: path → `<REPO_ROOT>` | `7230dc80096d` | `26ee4f8bd30c` |
| `research/results/raw/c15r2/logs/pins_start.log` | 1 | record: path → `<REPO_ROOT>` | `ff30517c0d24` | `4da8e8e570c7` |
| `research/results/raw/c15r2/logs/pre_fix_02201df/pins_start.log` | 1 | record: path → `<REPO_ROOT>` | `62cd0349d557` | `3bf236c4cc07` |
| `research/results/raw/c15r2/logs/pre_fix_02201df/smoke2_qwen3-1.7b.log` | 2 | record: path → `<REPO_ROOT>` | `d0324e01903c` | `af4b6c89e05d` |
| `research/results/raw/c15r2/logs/pre_fix_02201df/smoke2_qwen3-4b.log` | 2 | record: path → `<REPO_ROOT>` | `fdaa2d4413cb` | `dffb69ef5e7f` |
| `research/results/raw/c15r2/logs/pre_fix_02201df/smoke2_qwen3.5-2b.log` | 2 | record: path → `<REPO_ROOT>` | `56bcd48fd654` | `ed39cf1e66f5` |
| `research/results/raw/c15r2/logs/pre_fix_02201df/z_qwen3-1.7b.log` | 1 | record: path → `<REPO_ROOT>` | `6b519e2e24e9` | `d2d320d7049e` |
| `research/results/raw/c15r2/logs/pre_fix_02201df/z_qwen3-4b.log` | 1 | record: path → `<REPO_ROOT>` | `f1e35ec751ee` | `5eff10fa66a9` |
| `research/results/raw/c15r2/logs/pre_fix_02201df/z_qwen3.5-2b.log` | 1 | record: path → `<REPO_ROOT>` | `318aaec9dcf4` | `b56776384085` |
| `research/results/raw/c15r2/logs/select2_qwen3-1.7b.log` | 2 | record: path → `<REPO_ROOT>` | `3689346e55a9` | `a43ae94cf023` |
| `research/results/raw/c15r2/logs/select2_qwen3-4b.killed_by_reboot_20261007T2200.log` | 1 | record: path → `<REPO_ROOT>` | `468e44c324d4` | `2961a0c608a4` |
| `research/results/raw/c15r2/logs/select2_qwen3-4b.log` | 2 | record: path → `<REPO_ROOT>` | `df41d52a3bb8` | `97ea2d484eb3` |
| `research/results/raw/c15r2/logs/select2_qwen3.5-2b.log` | 2 | record: path → `<REPO_ROOT>` | `1199760fd8b8` | `3f0f9601fa7b` |
| `research/results/raw/c15r2/logs/smoke2_qwen3-1.7b.log` | 2 | record: path → `<REPO_ROOT>` | `4b48f6af717d` | `e0317d7d3446` |
| `research/results/raw/c15r2/logs/smoke2_qwen3-4b.log` | 2 | record: path → `<REPO_ROOT>` | `518e30984ceb` | `3f32299b0835` |
| `research/results/raw/c15r2/logs/smoke2_qwen3.5-2b.log` | 2 | record: path → `<REPO_ROOT>` | `948b23c1c75d` | `43f3d7ffbafc` |
| `research/results/raw/c15r2/logs/z_qwen3-1.7b.log` | 1 | record: path → `<REPO_ROOT>` | `336a6da8e464` | `11ed53e43fe1` |
| `research/results/raw/c15r2/logs/z_qwen3-4b.log` | 1 | record: path → `<REPO_ROOT>` | `867c68507052` | `cc56eb3fc572` |
| `research/results/raw/c15r2/logs/z_qwen3.5-2b.log` | 1 | record: path → `<REPO_ROOT>` | `7048e87e0a84` | `bd00892bcf79` |
| `research/results/raw/c15r2/prerun/pytest_r2.log` | 1 | record: path → `<REPO_ROOT>` | `2e7cbf59bdc0` | `a043a87b26fb` |
| `research/results/raw/c15r2/prerun/pytest_r2_engfix.log` | 1 | record: path → `<REPO_ROOT>` | `dcdf41dfb5b3` | `83f06fdcce94` |
| `research/results/raw/calibration/log_C4.txt` | 1 | record: path → `<REPO_ROOT>` | `9b6c134cb13f` | `c746d91967c3` |
| `research/results/raw/calibration/log_chain.txt` | 1 | record: path → `<REPO_ROOT>` | `771a02e69897` | `f7444ff7c43c` |
| `research/results/raw/calibration/log_d2_S3_9102.txt` | 8 | record: path → `<REPO_ROOT>` | `f05ccf95578f` | `20facec23f54` |
