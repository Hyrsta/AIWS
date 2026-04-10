# Session State

- Current objective: `flash_attn` is fixed on `RXL`, and the helper launcher now defaults to `flash_attn` on the RTX A6000 after a full successful validation run.
- Last confirmed decision: Leonardo asked to continue past the basic sanity check and verify why SAM3D still chose `sdpa`.
- Blocker or open question: No active blocker on the flash-attn path; the remaining choice is whether to keep the launcher-only override or later patch the repo-level GPU whitelist.
- Next useful move: Report that the A6000 runtime now works with `flash_attn` via the launcher default, and only patch repo code if Leonardo wants that behavior inside the codebase itself.
