# ops

- When a conda env on this machine is activated from a strict bash wrapper, avoid `set -u` during `source conda.sh` and `conda activate`; some activate scripts assume unset tool variables like `ADDR2LINE` are allowed.
- For destructive cleanup of local agent state, prefer moving the old copy to Trash over permanent deletion.
- On `RXL`, the SAM3D repo only auto-enables `flash_attn` for A100/H100/H200. If the RTX A6000 env already has a working local `flash_attn` build, prefer a launcher-level default of `ATTN_BACKEND=flash_attn` and `SPARSE_ATTN_BACKEND=flash_attn` instead of patching repo code first.
- When `torch.hub` on `RXL` tries to revalidate `facebookresearch/dinov2` against GitHub, patch the runner to redirect that load to `/home/rxl/.cache/torch/hub/facebookresearch_dinov2_main` with `source="local"` rather than relying on outbound network access.
