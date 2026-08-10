# Main Algorithm Code Screenshots Design

## Objective

Produce a standalone folder of five publication-ready PNG screenshots that explain the implemented algorithm path from SAM3 mask generation through SAM3D mesh reconstruction and into Cadrille CAD generation. Do not modify any Word document.

## Source and provenance

- Capture only code from the live RXL server.
- Record the source repository, file path, line range, and Git commit for every screenshot.
- Use the live AIWS checkout at commit `efe8dc6` for SAM3D, bridge, Cadrille, and end-to-end orchestration excerpts.
- Use the separate `/ssd1/rxl/zhankaiming/MV-SAM3D` checkout for SAM3 mask-generation code and record its independently verified commit.
- Do not imply that the current AIWS production entry point invokes SAM3 directly. Its verified entry point accepts an image and mask, then runs SAM3D and Cadrille.

## Deliverables

Create an ordered folder containing:

1. `01_sam3_mask_generation.png`
2. `02_sam3d_mesh_reconstruction.png`
3. `03_sam3d_to_cadrille_bridge.png`
4. `04_cadrille_cad_generation.png`
5. `05_end_to_end_pipeline_orchestration.png`

## Content selection

### 1. SAM3 mask generation

Show the smallest coherent excerpt that initializes or calls SAM3 and produces the object mask used by later reconstruction. Source from `MV-SAM3D/preprocessing/sam3_segmenter.py` unless live inspection identifies a more authoritative entry point.

### 2. SAM3D mesh reconstruction

Show image and mask loading, mask validation, SAM3D inference, and mesh materialization. Prefer the current GUI job implementation in `AIWS/gui/backend/simple_reconstruct_job.py` because it is the verified user-facing path.

### 3. SAM3D-to-Cadrille bridge

Show the conversion of SAM3D output into the Cadrille split/input representation. Prefer `AIWS/scripts/sam3d_cadrille_bridge.py` or the bridge call in the end-to-end runner, whichever provides the clearest coherent data transition.

### 4. Cadrille CAD generation

Show the core Cadrille inference path that converts the prepared point-cloud or image representation into parametric CAD code and materialized CAD output. Use current upstream/deployed Cadrille code plus the minimal AIWS wrapper context needed for correctness.

### 5. End-to-end orchestration

Show the stage sequence that runs SAM3D, prepares the bridge, launches Cadrille, and selects the resulting mesh/code/BRep. Prefer the verified orchestration in `AIWS/scripts/e2e_sam3d_to_cadrille.py` or `AIWS/gui/backend/simple_reconstruct_job.py`.

## Visual treatment

- PNG format, approximately 1920 pixels wide at 2x rendering scale.
- Clean light editor theme suitable for a Chinese technical report and printing.
- Preserve syntax highlighting, indentation, and visible line numbers.
- Include a restrained header containing the stage name, source path, line range, and short commit.
- Limit each image to roughly 20-30 code lines when possible; split only if a single coherent operation cannot fit legibly.
- Exclude terminal chrome, prompts, unrelated imports, secrets, credentials, host configuration, and absolute paths that do not help explain the algorithm.
- Do not annotate or rewrite the code inside the screenshot. Short ellipsis separators are allowed only where omitted lines do not change control flow.

## Verification

- Confirm every excerpt against the current RXL files immediately before rendering.
- Verify that all five PNGs open successfully at full resolution.
- Inspect every image for readable text, correct cropping, intact line numbers, no clipped code, and consistent styling.
- Cross-check that the five images form the intended narrative: SAM3 mask -> SAM3D mesh -> bridge -> Cadrille CAD -> end-to-end orchestration.
- Keep the folder limited to the five requested PNG deliverables.
