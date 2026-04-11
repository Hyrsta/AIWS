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
- [19:32] When explaining AIWS online/offline structure, explain what the online pipeline is used for and what the offline pipeline is used for before listing the model stacks
  Type: documentation
  Context: Leonardo rejected a model-first phrasing in the docs and asked for purpose-first explanation
  Confirmed: yes
- [22:03] Avoid overloaded parenthetical section titles like `Project structure (AIWS online/offline split and the offline CAD pipeline)`; prefer short headings such as `AIWS project structure`, followed by a plain direct opening sentence
  Type: documentation
  Context: Leonardo explicitly called out the heading and opening sentence as bad wording
  Confirmed: yes
- [22:04] For AIWS scripts, do not use `run_...`-style filenames for module workflows; prefer module-first names such as `sam3d_*` and `cadrille_*`
  Type: naming
  Context: Leonardo asked for more consistent script naming based on module names
  Confirmed: yes
- [22:07] In AIWS project-structure sections, write the current structure directly: online pipeline, offline pipeline, current repo layout, and upstream/submodule vs AIWS wrapper responsibilities. Avoid padded headings and repetitive focus sentences.
  Type: documentation
  Context: Leonardo asked to rewrite the project-structure section using the current structure and check the whole documentation for similar wording problems
  Confirmed: yes
- [19:38] In the SOP structure, SAM3D and Cadrille should be framed as parallel modules. If there is `SAM3D Environment Setup` and `SAM3D Inference Workflow`, there should also be `Cadrille Environment Setup` and `Cadrille Inference Workflow`
  Type: documentation
  Context: Leonardo rejected the old `Continuing from SAM3D to Cadrille` framing because it made Cadrille look subordinate instead of parallel
  Confirmed: yes
- [19:42] When Leonardo provides historical setup notes, fold them into the docs but rewrite them to match the current workspace paths, terminology, and current operational framing instead of copying the old standalone wording
  Type: documentation
  Context: Leonardo pointed to `Cadrille Env & Run Cmd.md` as history for the Cadrille environment and asked that it be adapted to the current AIWS workspace
  Confirmed: yes
- [19:48] In the SOP, the single end-to-end SAM3D-to-Cadrille bridge script is orchestration and should live in its own bridge/orchestration section, not inside the Cadrille inference section
  Type: documentation
  Context: Leonardo rejected placing `6.1 Single end-to-end bridge script` under Cadrille because it spans both modules
  Confirmed: yes
- [19:52] Do not use meta wording like `Recommended future configuration for ...` in these technical SOP sections. Matching module sections should mirror each other structurally, like Part 4 SAM3D and Part 6 Cadrille
  Type: documentation
  Context: Leonardo rejected the meta phrasing in Part 6 and explicitly asked to remember not to do this again
  Confirmed: yes
- [19:54] When Section 7 covers the cross-module bridge, explicitly include `end-to-end` in the heading because that framing matters for professor-facing communication
  Type: documentation
  Context: Leonardo said the professor wants to hear the `end-to-end` wording in this section title
  Confirmed: yes
- [20:33] For AIWS integration with official upstream repos, keep upstream model/runtime code untouched whenever possible; put only HF id or local-path handling, argument translation, and GPU-memory logging in thin wrappers instead of copying model files
  Type: workflow
  Context: Leonardo corrected the Cadrille cleanup approach and asked to keep only the minimum wrapper layer, removing the copied model helper once official `cadrille.py` was verified to work
  Confirmed: yes
- [20:36] For wrapper script naming, do not prefix AIWS helpers with `aiws_`; use module-first names like `cadrille_*` / `sam3d_*`, and let the cross-module e2e script keep a combined name
  Type: workflow
  Context: Leonardo asked to make the naming convention between Cadrille and SAM3D consistent and explicitly rejected the `aiws_` prefix for script names
  Confirmed: yes
- [20:43] Do not use symlinks as the steady-state cleanup for this AIWS integration; stop in-flight runs and relaunch from the cleaned final layout instead
  Type: workflow
  Context: Leonardo rejected my temporary symlink compatibility approach and asked for a clean stop-and-rerun instead
  Confirmed: yes
