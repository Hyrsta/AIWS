# Proactivity Log

- 2026-04-10: Initialized local proactive state for the welding-algorithm workspace.
- 2026-04-10: Started tracking the RXL flash-attn build as an active follow-up item.
- 2026-04-10: Leonardo corrected the storage location. Keep proactive state in `proactivity/` inside this workspace.
- 2026-04-10: The forced local `flash_attn` build on `RXL` completed, and `import flash_attn` now succeeds in the `sam3d-objects` env.
- 2026-04-10: Moved the old `~/proactivity` copy to Trash after the workspace-local state was confirmed.
- 2026-04-10: Patched `/ssd1/rxl/zhankaiming/run_sam3d_demo.sh` to disable `set -u` during `conda activate`, then reran SAM3D successfully end to end.
- 2026-04-10: Confirmed the repo only auto-enables `flash_attn` on A100/H100/H200; on the RTX A6000 it stayed on `sdpa` unless `ATTN_BACKEND=flash_attn` and `SPARSE_ATTN_BACKEND=flash_attn` were set.
- 2026-04-10: Verified that the A6000 can run SAM3D successfully with `flash_attn`, then updated the helper launcher to default those backend env vars so future runs use `flash_attn` automatically.
