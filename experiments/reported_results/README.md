# Reproduce the reported-results presentation

`generate.py` converts the committed reported-summary JSON into LaTeX tables,
numerical macros, a CSV, two PDF figures, PNG previews, and a checksum manifest.
It does **not** launch training or reconstruct missing raw measurements.

```sh
python3 -m pip install -r experiments/reported_results/requirements.txt
python3 experiments/reported_results/generate.py
python3 experiments/reported_results/generate.py --check
```

Inputs and outputs are under
`artifacts/reported_results/2026-09-22/`. See that bundle's `README.md` for
provenance, interpretive limits, and Overleaf integration. Both figures are
generated from the JSON in one pass; no hand-edited SVG, PDF, or table cells are
needed. The source snapshots and any additional files in the bundle are included
in the manifest when generating. Regenerate the manifest after adding sources.

Validated environment: Python 3.9.6, Matplotlib 3.9.4, NumPy 2.0.2. The generator
uses the noninteractive Agg backend and temporary writable font caches.
PDFs use embedded TrueType fonts, no runtime LaTeX, and no creation timestamps.
The paper uses the committed PDFs, so Overleaf needs no Python dependencies.
