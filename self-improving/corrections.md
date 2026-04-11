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
- [16:14] For dataset explanations in these AIWS report docs, describe only the unified `aiws5.2-usable` structure and avoid mentioning legacy intermediate layouts or symlink implementation details
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
- [16:44] In supervisor-facing run summaries, avoid internal variable names and present grouped metrics directly by dataset version and workpiece type when that is the practical question
  Type: documentation
  Context: Leonardo asked for clearer run-summary wording and a `V1/V2/NEW × workpiece` breakdown of runtime and memory
  Confirmed: yes
- [16:47] In dataset sections, place the conceptual explanation of `misc/` in the core-semantics/structure part, and keep only the current counts in the later current-condition part
  Type: documentation
  Context: Leonardo asked for a cleaner separation between dataset structure and current dataset status
  Confirmed: yes
- [16:54] For supervisor-facing summary tables, prefer merged-row style where it improves readability, round displayed values to two decimals, and use easier units like GB or hours when large raw numbers are distracting
  Type: documentation
  Context: Leonardo corrected the readability of the run-summary tables
  Confirmed: yes
- [16:56] Avoid repetitive or overly defensive wording around `misc/`; prefer a short neutral explanation such as the main benchmark using the cleaner single-instance portion while `misc/` stores separate cases, currently mainly multi-instance samples
  Type: documentation
  Context: Leonardo rejected the previous misc wording as out of place and repetitive
  Confirmed: yes
- [17:01] In grouped supervisor tables, merge repeated dataset-version rows when possible and remove empty zero-sample rows instead of listing placeholders
  Type: documentation
  Context: Leonardo corrected the table layout for the `V1/V2/NEW × workpiece` summary
  Confirmed: yes
- [17:05] In dataset explanations, keep statements like "misc is mainly multi-instance" only in the current-condition block, not in the core-structure block
  Type: documentation
  Context: Leonardo corrected the placement of current-condition language in the AIWS docs
  Confirmed: yes
- [17:07] For Cadrille quality metrics in the report, follow the paper convention: IoU in percent and Chamfer Distance multiplied by 10^3, instead of raw small decimals
  Type: documentation
  Context: Leonardo asked the report to match the original Cadrille paper’s metric convention
  Confirmed: yes
- [17:31] In Cadrille reporting, use the successful-output basis only for runtime/throughput summaries, and explicitly note that PC is slower partly because it generates 5 candidates per input (`n_samples=5`)
  Type: documentation
  Context: Leonardo corrected the reporting basis and asked the report to explain the PC-vs-IMG runtime difference more directly
  Confirmed: yes
- [19:13] For AIWS project framing, say AIWS is split into online and offline pipelines, and this CAD reconstruction pipeline is the offline pipeline
  Type: documentation
  Context: Leonardo corrected the project-structure wording after I overstated AIWS as if it were just the offline layer
  Confirmed: yes
- [19:17] In AIWS offline-pipeline wording, describe it as the offline CAD reconstruction pipeline with two stages, "RGB image → mesh reconstruction" and "mesh reconstruction → CAD reconstruction", rather than collapsing it into "RGB image → Cadrille"
  Type: documentation
  Context: Leonardo corrected the pipeline wording to match the provided project framing slide
  Confirmed: yes
- [19:18] Never write meta lead-ins like "The correct framing is" or similar in project docs; write directly in Leonardo's voice as if he authored the document
  Type: documentation
  Context: Leonardo corrected the report tone again and asked that docs avoid assistant/meta setup sentences entirely
  Confirmed: yes
- [19:20] When introducing AIWS project structure, explicitly explain both halves: the online vision pipeline is `YOLOv11-seg + GenPose++ + FoundationPose`, and the offline CAD reconstruction pipeline is `SAM3D + Cadrille`
  Type: documentation
  Context: Leonardo asked the docs to explain the online/offline split with the actual model stacks, not just the words "online" and "offline"
  Confirmed: yes
- [19:24] For Cadrille framing, say it starts from the reconstructed mesh, then either samples point clouds from that mesh for PC mode or renders 4-view RGB images from that mesh for IMG mode before CAD reconstruction
  Type: documentation
  Context: Leonardo corrected the Cadrille description and pointed to `Cadrille 方法研究.pptx` as the reference framing
  Confirmed: yes
