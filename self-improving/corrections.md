# Corrections Log

## 2026-04-10
- [14:23] Changed proactivity state location from `~/proactivity/` to `proactivity/`
  Type: technical
  Context: workspace-local skill initialization
  Confirmed: yes
- [14:23] Initialize `self-improving/` alongside `proactivity/` inside the workspace
  Type: workflow
  Context: Leonardo corrected the setup approach
  Confirmed: yes
- [23:39] For bilingual documentation edits, always update both English and Chinese versions together
  Type: documentation
  Context: Leonardo requested synchronized EN+ZH edits whenever one side is requested
  Confirmed: yes
- [23:49] Keep evaluation aligned with paper method; do not switch to voxel IoU by default
  Type: evaluation
  Context: Leonardo rejected changing evaluation method and asked to follow paper metrics
  Confirmed: yes
- [23:53] Use Cadrille pipeline selection (evaluate.py best_names), not fixed candidate index by default
  Type: evaluation
  Context: Leonardo called out index-0 selection as not meaningful and requested paper-aligned pipeline
  Confirmed: yes
- [23:53] Keep modality reporting split between pc and img runs
  Type: experiment-design
  Context: Leonardo requested explicit modality separation
  Confirmed: yes
- [00:11] For Cadrille generation count, use img=1 and pc=5 by default
  Type: experiment-design
  Context: Leonardo clarified deterministic image input should use one generation, while point-cloud branch keeps multi-sample search
  Confirmed: yes
- [16:11] In generated project documents, use the human author name 詹铠铭 / Kaiming Zhan and omit Audience fields unless explicitly needed
  Type: documentation
  Context: Leonardo corrected the author metadata and asked to remove audience lines from the generated report materials
  Confirmed: yes
