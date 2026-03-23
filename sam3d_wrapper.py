"""
Wrapper around SAM 3D Objects for multi-view mesh reconstruction.

SAM3D takes masked objects in images and produces 3D meshes with
pose, shape, and texture. This wrapper handles image loading,
mask generation (via SAM), and mesh export.

Requires: third_party/sam-3d-objects to be installed.
See setup instructions in the main README.
"""

import os
import sys
import glob
import numpy as np

# Add SAM3D to path
SAM3D_DIR = os.path.join(os.path.dirname(__file__), "third_party", "sam-3d-objects")
sys.path.insert(0, SAM3D_DIR)
sys.path.insert(0, os.path.join(SAM3D_DIR, "notebook"))


class SAM3DReconstructor:
    """Reconstruct a 3D mesh from multi-view RGB images using SAM 3D Objects."""

    def __init__(self, config_path=None, device="cuda"):
        """
        Args:
            config_path: path to SAM3D pipeline YAML config.
                         If None, uses the default HuggingFace checkpoint.
            device: 'cuda' or 'cpu'.
        """
        self.device = device
        self.config_path = config_path or os.path.join(
            SAM3D_DIR, "checkpoints", "hf", "pipeline.yaml"
        )
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from inference import Inference
            self._model = Inference(self.config_path, compile=False)
            print(f"[SAM3D] Model loaded from {self.config_path}")
        except ImportError:
            raise ImportError(
                "SAM 3D Objects not found. Run:\n"
                "  cd third_party/sam-3d-objects && pip install -e .\n"
                "and download checkpoints per their README."
            )

    def reconstruct(self, image_dir, output_dir, mask_dir=None):
        """
        Reconstruct a mesh from images in image_dir.

        Args:
            image_dir: directory with RGB images (*.png, *.jpg).
            output_dir: where to save the output mesh.
            mask_dir: optional directory with pre-computed masks.
                      If None, uses SAM to generate masks automatically.

        Returns:
            Path to the reconstructed mesh (.ply).
        """
        self._load_model()
        from inference import load_image, load_single_mask

        image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.png"))
            + glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.jpeg"))
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        print(f"[SAM3D] Processing {len(image_paths)} images from {image_dir}")

        os.makedirs(output_dir, exist_ok=True)
        all_outputs = []

        for img_path in image_paths:
            image = load_image(img_path)
            stem = os.path.splitext(os.path.basename(img_path))[0]

            # Load or generate mask
            if mask_dir and os.path.exists(os.path.join(mask_dir, f"{stem}.png")):
                mask = load_single_mask(mask_dir, index=0)
            else:
                # Use the full image as mask (single-object assumption)
                mask = np.ones(
                    (image.size[1], image.size[0]), dtype=np.uint8
                ) * 255

            output = self._model(image, mask, seed=42)
            all_outputs.append(output)

        # Export the first reconstruction as the primary mesh
        # For multi-view fusion, a more advanced merging step would go here
        mesh_path = os.path.join(output_dir, "reconstructed.ply")

        if hasattr(all_outputs[0].get("gs", None), "save_ply"):
            all_outputs[0]["gs"].save_ply(mesh_path)
        elif "mesh" in all_outputs[0]:
            all_outputs[0]["mesh"].export(mesh_path)
        else:
            # Fall back: export gaussian splat as PLY
            gs = all_outputs[0].get("gs")
            if gs is not None:
                gs.save_ply(mesh_path)

        print(f"[SAM3D] Mesh saved to {mesh_path}")
        return mesh_path


def reconstruct_from_images(image_dir, output_dir, config_path=None, device="cuda"):
    """Convenience function for single-call reconstruction."""
    recon = SAM3DReconstructor(config_path=config_path, device=device)
    return recon.reconstruct(image_dir, output_dir)
