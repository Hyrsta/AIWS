# Working Buffer

- RXL environment: `sam3d-objects`
- flash-attn target: local source build for `sm80` only
- Known bad path: prebuilt wheels require `GLIBC_2.32`
- Verification after build: `import flash_attn`, then a quick SAM3D sanity check
