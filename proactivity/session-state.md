# Session State

- Current objective: Keep Cadrille on the official upstream repo by moving AIWS-specific inference, conversion, evaluation, and GPU-memory logging into wrapper scripts, then rerun the full PC and IMG experiments on RXL against the official clone.
- Last confirmed decision: Leonardo wants the same pattern as SAM3D, official repo code stays clean and AIWS-only behavior lives in standalone wrapper files. After rechecking the diffs, only a thin test wrapper is needed for processor-path/HF-id override, sample-count and batch-size control, and GPU-memory logging, and the copied model helper has been removed.
- Blocker or open question: The clean rerun is active on RXL and the docs/report update still depends on the final shard outputs and logged peaks.
- Next useful move: Monitor `/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-clean-rerun-20260411-204657`, then aggregate the resulting `gpu_memory.json` files and update the English and Chinese report sections together. The old modified Cadrille repo remains archived under `AIWS/backups/`, and the temporary symlink-based compatibility path has been removed.
