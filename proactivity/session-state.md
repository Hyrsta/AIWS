# Session State

- Current objective: Keep Cadrille on the official upstream repo by moving AIWS-specific inference, conversion, evaluation, and GPU-memory logging into wrapper scripts, then rerun the full PC and IMG experiments on RXL against the official clone.
- Last confirmed decision: Leonardo wants the same pattern as SAM3D, official repo code stays clean and AIWS-only behavior lives in standalone wrapper files.
- Blocker or open question: The wrapper-based full rerun is now active on RXL and the docs/report update still depends on the final shard outputs and logged peaks.
- Next useful move: Monitor `/ssd1/rxl/zhankaiming/AIWS/outputs/cadrille-official-wrapper-rerun-20260411-202252`, then aggregate the resulting `gpu_memory.json` files and update the English and Chinese report sections together.
