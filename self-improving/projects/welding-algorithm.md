# welding-algorithm

- Keep workspace-local skill state in `proactivity/` and `self-improving/`, not in `~/`.
- Use this project workspace as the home for durable agent operating notes.
- For evaluation reporting, follow the paper-aligned method (Cadrille `evaluate.py` Chamfer Distance + mesh IoU) unless Leonardo explicitly requests an alternative.
- For Cadrille experiments, separate modality runs explicitly (`mode=pc` and `mode=img`) instead of mixing them in one metric report.
- For Cadrille generation count defaults, use `n_samples=5` for `mode=pc` and `n_samples=1` for `mode=img`.
