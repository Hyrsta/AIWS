"""Fast tests for saved Cadrille IMG input artifacts."""
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

if "torch" not in sys.modules:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.is_tensor = lambda value: False
    torch.bfloat16 = object()
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.ConcatDataset = object
    torch.utils.data.DataLoader = object
    sys.modules["torch"] = torch
    sys.modules["torch.utils"] = torch.utils
    sys.modules["torch.utils.data"] = torch.utils.data

if "tqdm" not in sys.modules:
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda value, *args, **kwargs: value
    sys.modules["tqdm"] = tqdm_mod

if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = object
    sys.modules["transformers"] = transformers

try:
    from PIL import Image
except Exception:  # pragma: no cover - the RXL env has Pillow
    Image = None

MOD = os.path.join(os.path.dirname(__file__), "..", "scripts", "cadrille_infer_wrapper.py")
spec = importlib.util.spec_from_file_location("cadrille_infer_wrapper", MOD)
ciw = importlib.util.module_from_spec(spec)
sys.modules["cadrille_infer_wrapper"] = ciw
spec.loader.exec_module(ciw)


def test_write_input_render_artifact_saves_img_batch_render():
    assert Image is not None
    with tempfile.TemporaryDirectory() as d:
        image = Image.new("RGB", (8, 8), (7, 11, 13))
        path = ciw.write_input_render_artifact(
            Path(d),
            mode="img",
            source_stem="part",
            output_file_name="part+0.py",
            generation_id=0,
            video=[image],
        )

        out = Path(d) / "input_renders" / "part+0.png"
        meta = Path(d) / "input_renders" / "part+0.json"
        assert path == str(out)
        assert out.exists()
        assert meta.exists()
        assert "batch['input_videos']" in meta.read_text(encoding="utf-8")


def test_collate_with_input_artifacts_preserves_raw_video_images():
    raw_video = ["render-grid"]

    def fake_collate(batch, processor, n_points, eval=False):
        assert processor == "processor"
        assert n_points == 256
        assert eval is True
        return {"file_name": [m["file_name"] for m in batch]}

    result = ciw.collate_with_input_artifacts(
        [{"file_name": "part", "video": raw_video}],
        upstream_collate=fake_collate,
        processor="processor",
        n_points=256,
        eval=True,
    )

    assert result["input_videos"] == [raw_video]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} IMG artifact tests passed.")


if __name__ == "__main__":
    _run_all()
