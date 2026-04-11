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
- [16:14] For dataset explanations in these AIWS report docs, describe only the unified `aiws5.2-usable` structure and avoid mentioning `aiws5.2-usable-split`, `materialized`, or symlink implementation details
  Type: documentation
  Context: Leonardo asked to simplify the dataset story for the docs and presentation
  Confirmed: yes
- [16:16] When explaining `misc/` in the AIWS docs, say it is excluded mainly because of multi-instance labels and note that it may be fixed and reused later
  Type: documentation
  Context: Leonardo asked to clarify why `misc/` is not used in the current formal pipeline
  Confirmed: yes
- [16:25] In supervisor-facing documents, remove assistant/meta narration and write directly in Leonardo's voice rather than using phrases like "for presentation purposes" or "can be summarized simply"
  Type: documentation
  Context: Leonardo corrected the report tone and asked that supervisor documents read as if written directly by him
  Confirmed: yes
- [16:27] In dataset explanations, include the current dataset condition, not just the folder structure: present workpiece coverage, depth availability, subset population, and notable empty/missing categories
  Type: documentation
  Context: Leonardo asked the AIWS docs to explain what currently exists in the dataset, not only how folders are organized
  Confirmed: yes
- [16:37] In markdown reports, nested bullet subpoints must be indented one level deeper so they render correctly in the exported PDF
  Type: documentation
  Context: Leonardo corrected the list formatting for supervisor-facing documents
  Confirmed: yes
- [16:39] Keep structure and current-condition statements clearly separated in dataset sections: folder semantics for `misc/` belong in the structure/core-semantics part, while counts like multi-instance and unannotated cases belong in the current-condition part
  Type: documentation
  Context: Leonardo corrected the organization of the AIWS dataset explanation
  Confirmed: yes
