# ops

- When a conda env on this machine is activated from a strict bash wrapper, avoid `set -u` during `source conda.sh` and `conda activate`; some activate scripts assume unset tool variables like `ADDR2LINE` are allowed.
- For destructive cleanup of local agent state, prefer moving the old copy to Trash over permanent deletion.
- On `RXL`, the SAM3D repo only auto-enables `flash_attn` for A100/H100/H200. If the RTX A6000 env already has a working local `flash_attn` build, prefer a launcher-level default of `ATTN_BACKEND=flash_attn` and `SPARSE_ATTN_BACKEND=flash_attn` instead of patching repo code first.
- When `torch.hub` on `RXL` tries to revalidate `facebookresearch/dinov2` against GitHub, patch the runner to redirect that load to `/home/rxl/.cache/torch/hub/facebookresearch_dinov2_main` with `source="local"` rather than relying on outbound network access.
- Before launching a long multi-instance SAM3D batch on `RXL`, do a real 1-instance smoke test with the exact runner and environment first, then add per-instance timing and GPU-memory metrics before scaling out.
- If multiple A6000 GPUs are available, prefer one process per GPU with deterministic task sharding (`CUDA_VISIBLE_DEVICES=<gpu>`, `--num-shards N`, `--shard-index i`) over a single-process serial run.
