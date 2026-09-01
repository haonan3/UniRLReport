# UniRL trajectory-IR CPU evidence

- Run ID: `cpu_f5d7104_macos_arm64`
- UniRL commit: `f5d710406b215bb7a0b387fdd37e4d4778b92338`
- Checkout clean: `True`
- Python: `3.12.14`
- PyTorch: `2.11.0`
- Platform: `macOS-26.3-arm64-arm-64bit`
- Timed device: `CPU` (1 thread)
- Protocol: 10 warm-up + 100 measured repetitions

## Correctness

All 10 executable checks passed. The process aborts on the first failed assertion.

- PASS — `cross_stage_lineage_validation`
- PASS — `split_concat_exact_roundtrip`
- PASS — `whole_tree_select_and_slice`
- PASS — `reward_propagation_and_group_advantages`
- PASS — `invalid_lineage_rejected`
- PASS — `primitive_modality_mismatch_rejected`
- PASS — `primitive_batch_mismatch_rejected`
- PASS — `packed_variable_length_roundtrip`
- PASS — `tensorref_view_selection_and_materialization`
- PASS — `tensorref_bounds_rejected`

## Timing summary

| Root groups | Leaf rows | Operation | Median (us) | p90 (us) |
|---:|---:|---|---:|---:|
| 8 | 64 | `sample.split` | 792.5 | 810.0 |
| 8 | 64 | `sample.split_concat` | 201.3 | 212.0 |
| 8 | 64 | `sample.select_half` | 954.2 | 977.7 |
| 8 | 64 | `sample.propagate_rewards` | 37.0 | 38.0 |
| 8 | 64 | `tensorref.select_contiguous` | 1.9 | 2.0 |
| 8 | 64 | `tensorref.select_strided` | 20.1 | 20.4 |
| 32 | 256 | `sample.split` | 3172.3 | 3248.5 |
| 32 | 256 | `sample.split_concat` | 488.6 | 496.8 |
| 32 | 256 | `sample.select_half` | 3538.6 | 3615.8 |
| 32 | 256 | `sample.propagate_rewards` | 76.0 | 78.8 |
| 32 | 256 | `tensorref.select_contiguous` | 1.8 | 1.9 |
| 32 | 256 | `tensorref.select_strided` | 74.1 | 76.8 |
| 128 | 1024 | `sample.split` | 12834.0 | 13091.6 |
| 128 | 1024 | `sample.split_concat` | 1653.0 | 1700.5 |
| 128 | 1024 | `sample.select_half` | 13893.5 | 14139.3 |
| 128 | 1024 | `sample.propagate_rewards` | 234.6 | 245.7 |
| 128 | 1024 | `tensorref.select_contiguous` | 2.0 | 2.1 |
| 128 | 1024 | `tensorref.select_strided` | 327.9 | 337.5 |

## Scope

These are synthetic, single-process CPU measurements of container and reference operations. They do not measure rollout throughput, learner throughput, network transport, GPU utilization, training quality, or synchronous/asynchronous convergence, and must not be extrapolated to those claims.
