# Trajectory-IR CPU evidence

This harness imports the pinned UniRL source tree and checks the trajectory container,
packed-segment, reward/advantage, and lazy tensor-reference invariants used by the paper.
It also records single-process CPU microbenchmarks for regression purposes. It does
not launch models or distributed workers and is not an end-to-end performance test.

From the `UniRLReport` repository root, using Python 3.12:

```bash
python -m venv /tmp/unirl-paper-venv
/tmp/unirl-paper-venv/bin/pip install -r experiments/trajectory_ir/requirements-cpu.txt
PYTHONPATH=../UniRL /tmp/unirl-paper-venv/bin/python \
  experiments/trajectory_ir/run_cpu_evidence.py \
  --unirl-root ../UniRL \
  --output-dir artifacts/trajectory_ir/cpu_f5d7104_macos_arm64 \
  --expected-commit f5d710406b215bb7a0b387fdd37e4d4778b92338 \
  --warmup 10 --repeats 100 --threads 1
```

The harness refuses a different commit or a dirty UniRL checkout by default. It emits
the environment manifest, individual correctness outcomes, raw timing trials, captured
stdout/stderr, a CSV summary, generated LaTeX rows/table, a Markdown report, and
SHA-256 checksums.
