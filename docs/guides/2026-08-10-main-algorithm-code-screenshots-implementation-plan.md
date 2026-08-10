# Main Algorithm Code Screenshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce five ordered, publication-ready PNG screenshots that accurately explain SAM3 mask generation, SAM3D mesh reconstruction, the SAM3D-to-Cadrille bridge, Cadrille CAD generation, and the end-to-end orchestration.

**Architecture:** Copy the six authoritative source files read-only from RXL into a task-local temporary directory, extract only the approved line ranges, and syntax-highlight them with Pygments. Wrap the highlighted excerpts in deterministic local HTML cards and rasterize them with headless Google Chrome at 2x scale; the workspace receives only the five final PNGs.

**Tech Stack:** SSH/SCP, Pygments 2.19.2 (`/Users/hyrsta/Library/Python/3.9/bin/pygmentize`), bundled Python 3 with Pillow 12.2.0, Playwright Chromium headless shell, HTML/CSS.

## Global Constraints

- Create exactly five PNG files under `Main Algorithm Code Screenshots/`; do not add a README, manifest, HTML, or source snapshot to that folder.
- Use only code copied from the live RXL server immediately before rendering.
- Use a clean light code-editor treatment, visible line numbers, a restrained stage/source/commit header, and an approximately 1920-pixel output width.
- Preserve code verbatim. Ellipsis separators may appear only between independently copied source ranges.
- Exclude terminal chrome, credentials, SSH configuration, checkpoint paths, and unrelated imports.
- State the SAM3 boundary honestly: SAM3 is shown from the separate MV-SAM3D repository; the current AIWS entry point consumes an image and an existing mask.
- The selected SAM3D excerpt in `simple_reconstruct_job.py` must match commit `efe8dc6` even though RXL has an unrelated working-tree change at lines 843-844.

---

### Task 1: Capture authoritative RXL sources and provenance

**Files:**
- Read remotely: `/ssd1/rxl/zhankaiming/MV-SAM3D/preprocessing/sam3_segmenter.py`
- Read remotely: `/ssd1/rxl/zhankaiming/AIWS/gui/backend/simple_reconstruct_job.py`
- Read remotely: `/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_cadrille_bridge.py`
- Read remotely: `/ssd1/rxl/zhankaiming/AIWS/repos/cadrille/test.py`
- Read remotely: `/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_evaluate_wrapper.py`
- Read remotely: `/ssd1/rxl/zhankaiming/AIWS/scripts/e2e_sam3d_to_cadrille.py`
- Create temporarily: `$SHOT_TMP/src/...` copies of the six files above

**Interfaces:**
- Consumes: SSH alias `RXL` resolving to the accessible server checkout.
- Produces: an absolute `$SHOT_TMP` containing immutable source snapshots plus verified commit labels `MV-SAM3D 203f7c0`, `AIWS efe8dc6`, and `Cadrille 338db11`.

- [ ] **Step 1: Create an isolated capture directory**

Run:

```bash
SHOT_TMP=$(mktemp -d /private/tmp/aiws-code-shots.XXXXXX)
mkdir -p "$SHOT_TMP/src/mv-sam3d/preprocessing" "$SHOT_TMP/src/aiws/gui/backend" \
  "$SHOT_TMP/src/aiws/scripts" "$SHOT_TMP/src/cadrille"
```

Expected: `test -d "$SHOT_TMP/src"` exits 0.

- [ ] **Step 2: Re-check live repository heads and selected-file cleanliness**

Run:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 RXL '
git -C /ssd1/rxl/zhankaiming/MV-SAM3D rev-parse --short HEAD
git -C /ssd1/rxl/zhankaiming/AIWS rev-parse --short HEAD
git -C /ssd1/rxl/zhankaiming/AIWS/repos/cadrille rev-parse --short HEAD
git -C /ssd1/rxl/zhankaiming/MV-SAM3D diff --quiet -- preprocessing/sam3_segmenter.py
git -C /ssd1/rxl/zhankaiming/AIWS diff --quiet -- scripts/sam3d_cadrille_bridge.py scripts/cadrille_evaluate_wrapper.py scripts/e2e_sam3d_to_cadrille.py
git -C /ssd1/rxl/zhankaiming/AIWS/repos/cadrille diff --quiet -- test.py
diff -u <(git -C /ssd1/rxl/zhankaiming/AIWS show HEAD:gui/backend/simple_reconstruct_job.py | sed -n "873,914p") <(sed -n "873,914p" /ssd1/rxl/zhankaiming/AIWS/gui/backend/simple_reconstruct_job.py)
'
```

Expected: the three heads print `203f7c0`, `efe8dc6`, and `338db11`; all comparisons exit 0 with no diff.

- [ ] **Step 3: Copy the six source files without altering RXL**

Run:

```bash
scp RXL:/ssd1/rxl/zhankaiming/MV-SAM3D/preprocessing/sam3_segmenter.py "$SHOT_TMP/src/mv-sam3d/preprocessing/"
scp RXL:/ssd1/rxl/zhankaiming/AIWS/gui/backend/simple_reconstruct_job.py "$SHOT_TMP/src/aiws/gui/backend/"
scp RXL:/ssd1/rxl/zhankaiming/AIWS/scripts/sam3d_cadrille_bridge.py "$SHOT_TMP/src/aiws/scripts/"
scp RXL:/ssd1/rxl/zhankaiming/AIWS/scripts/cadrille_evaluate_wrapper.py "$SHOT_TMP/src/aiws/scripts/"
scp RXL:/ssd1/rxl/zhankaiming/AIWS/scripts/e2e_sam3d_to_cadrille.py "$SHOT_TMP/src/aiws/scripts/"
scp RXL:/ssd1/rxl/zhankaiming/AIWS/repos/cadrille/test.py "$SHOT_TMP/src/cadrille/"
```

Expected: `find "$SHOT_TMP/src" -type f | wc -l` prints `6`.

- [ ] **Step 4: Verify the copied excerpts contain the intended calls**

Run:

```bash
sed -n '94,120p' "$SHOT_TMP/src/mv-sam3d/preprocessing/sam3_segmenter.py" | rg 'set_image|set_text_prompt|masks|scores|best_mask'
sed -n '873,888p;897,914p' "$SHOT_TMP/src/aiws/gui/backend/simple_reconstruct_job.py" | rg 'input_mask|Inference|inference\(|mesh.export'
sed -n '139,170p' "$SHOT_TMP/src/aiws/scripts/sam3d_cadrille_bridge.py" | rg 'prepare_cadrille_split|stl_path|normalize_stl|copy2'
sed -n '58,77p' "$SHOT_TMP/src/cadrille/test.py" | rg 'model.generate|batch_decode|py_string'
sed -n '55,65p' "$SHOT_TMP/src/aiws/scripts/cadrille_evaluate_wrapper.py" | rg 'exec\(|compound_to_mesh|export'
sed -n '167,192p;203,236p;290p' "$SHOT_TMP/src/aiws/scripts/e2e_sam3d_to_cadrille.py" | rg 'sam_cmd|prepare_cadrille_split|run_cadrille_on_split|run_cmd'
```

Expected: every command prints all named pipeline calls and exits 0.

---

### Task 2: Render the five deterministic PNG screenshots

**Files:**
- Create temporarily: `$SHOT_TMP/render_screenshots.py`
- Create temporarily: `$SHOT_TMP/html/01.html` through `$SHOT_TMP/html/05.html`
- Create: `Main Algorithm Code Screenshots/01_sam3_mask_generation.png`
- Create: `Main Algorithm Code Screenshots/02_sam3d_mesh_reconstruction.png`
- Create: `Main Algorithm Code Screenshots/03_sam3d_to_cadrille_bridge.png`
- Create: `Main Algorithm Code Screenshots/04_cadrille_cad_generation.png`
- Create: `Main Algorithm Code Screenshots/05_end_to_end_pipeline_orchestration.png`

**Interfaces:**
- Consumes: `$SHOT_TMP/src`, exact ranges and commit labels defined below.
- Produces: five 1920-pixel-wide PNGs with syntax-highlighted code, line numbers, and source provenance.

- [ ] **Step 1: Define the five screenshot specifications**

Implement the following exact mapping in `render_screenshots.py`:

```python
SHOTS = [
    ("01_sam3_mask_generation.png", "SAM3 - Multi-view mask generation", [
        ("src/mv-sam3d/preprocessing/sam3_segmenter.py", 94, 120,
         "MV-SAM3D/preprocessing/sam3_segmenter.py", "203f7c0"),
    ]),
    ("02_sam3d_mesh_reconstruction.png", "SAM3D - Image and mask to mesh", [
        ("src/aiws/gui/backend/simple_reconstruct_job.py", 873, 888,
         "AIWS/gui/backend/simple_reconstruct_job.py", "efe8dc6"),
        ("src/aiws/gui/backend/simple_reconstruct_job.py", 897, 914,
         "AIWS/gui/backend/simple_reconstruct_job.py", "efe8dc6"),
    ]),
    ("03_sam3d_to_cadrille_bridge.png", "Bridge - SAM3D mesh to Cadrille input", [
        ("src/aiws/scripts/sam3d_cadrille_bridge.py", 139, 170,
         "AIWS/scripts/sam3d_cadrille_bridge.py", "efe8dc6"),
    ]),
    ("04_cadrille_cad_generation.png", "Cadrille - Generate and materialize CAD", [
        ("src/cadrille/test.py", 58, 77, "Cadrille/test.py", "338db11"),
        ("src/aiws/scripts/cadrille_evaluate_wrapper.py", 55, 65,
         "AIWS/scripts/cadrille_evaluate_wrapper.py", "efe8dc6"),
    ]),
    ("05_end_to_end_pipeline_orchestration.png", "End-to-end - SAM3D to Cadrille", [
        ("src/aiws/scripts/e2e_sam3d_to_cadrille.py", 167, 192,
         "AIWS/scripts/e2e_sam3d_to_cadrille.py", "efe8dc6"),
        ("src/aiws/scripts/e2e_sam3d_to_cadrille.py", 203, 236,
         "AIWS/scripts/e2e_sam3d_to_cadrille.py", "efe8dc6"),
        ("src/aiws/scripts/e2e_sam3d_to_cadrille.py", 290, 290,
         "AIWS/scripts/e2e_sam3d_to_cadrille.py", "efe8dc6"),
    ]),
]
```

Expected: the mapping contains five unique output names and no source range outside the six copied files.

- [ ] **Step 2: Generate syntax-highlighted HTML cards**

Implement `render_screenshots.py` so it:

```python
def source_range(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1:end]) + "\n"

def pygments_html(code: str, start: int) -> str:
    result = subprocess.run(
        [PYGMENTIZE, "-l", "python", "-f", "html",
         "-O", f"linenos=table,linenostart={start}"],
        input=code, text=True, capture_output=True, check=True,
    )
    return result.stdout
```

The generated HTML must use `Menlo, "PingFang SC", monospace`, a white page background, a pale-gray code card, 15 CSS-pixel code text, 1.55 line height, a 24 CSS-pixel stage title, a 13 CSS-pixel provenance label, 28 CSS-pixel outer padding, and a fixed 960 CSS-pixel viewport width. Insert a centered `omitted source lines` separator between non-contiguous ranges.

Expected: five UTF-8 HTML files exist and each contains a `<table class="highlighttable">`, its stage title, its source path, and its commit.

- [ ] **Step 3: Rasterize every card at 2x scale**

Have `render_screenshots.py` run the dedicated Chromium headless shell once per HTML file with:

```python
subprocess.run(
    [
        "/Users/hyrsta/Library/Caches/ms-playwright/chromium_headless_shell-1148/chrome-mac/headless_shell",
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--window-size=960,{viewport_height}",
        f"--screenshot={output_png}",
        input_html.resolve().as_uri(),
    ],
    check=True,
)
```

For each file, set the CSS body/card height and Chrome window height to the measured content height before the final capture so the PNG contains 28 CSS pixels of bottom padding and no large blank tail.

Expected: exactly five non-empty PNG files exist in `Main Algorithm Code Screenshots/`.

---

### Task 3: Validate visual and structural quality

**Files:**
- Verify: `Main Algorithm Code Screenshots/*.png`

**Interfaces:**
- Consumes: the five rendered PNGs.
- Produces: a final folder that passes size, count, filename, readability, cropping, and provenance checks.

- [ ] **Step 1: Run automated artifact checks**

Run with bundled Python:

```python
from pathlib import Path
from PIL import Image

root = Path("Main Algorithm Code Screenshots")
expected = [
    "01_sam3_mask_generation.png",
    "02_sam3d_mesh_reconstruction.png",
    "03_sam3d_to_cadrille_bridge.png",
    "04_cadrille_cad_generation.png",
    "05_end_to_end_pipeline_orchestration.png",
]
files = sorted(p.name for p in root.iterdir())
assert files == expected, files
for name in expected:
    with Image.open(root / name) as image:
        assert image.format == "PNG"
        assert image.width == 1920, (name, image.size)
        assert image.height >= 900, (name, image.size)
        image.verify()
```

Expected: the script exits 0 and reports no assertion.

- [ ] **Step 2: Inspect all five PNGs at original resolution**

Open each PNG with the local image viewer and verify:

- every code line and line number is readable;
- Chinese comments render as Chinese glyphs rather than boxes or blanks;
- no horizontal or vertical cropping occurs;
- omitted-line separators occur only between approved source ranges;
- stage, source path, and commit labels are consistent;
- the Cadrille image visibly includes both model generation and CAD materialization;
- the end-to-end image visibly includes SAM3D launch, bridge preparation, Cadrille runner creation, and final execution.

Expected: all five images pass without layout defects. If one fails, change only the temporary renderer/CSS, rerender all five, and repeat Tasks 3.1-3.2.

- [ ] **Step 3: Confirm repository scope and commit the deliverables**

Run:

```bash
git status --short
git add -- 'Main Algorithm Code Screenshots/'
git diff --cached --check
git commit -m 'docs: add main algorithm code screenshots'
```

Expected: the commit contains the five PNGs and no unrelated workspace files.
