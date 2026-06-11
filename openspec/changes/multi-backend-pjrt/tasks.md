# Tasks — multi-backend-pjrt

## 1. Plugin-agnostic loader

- [x] 1.1 `PJRT_PLUGIN_PATH` env (before legacy `LIBTPU_PATH`), API version capture, function-table size guard, `tpu_api_version()` accessor
- [x] 1.2 `cpp_plugin_probe` smoke test: version + devices + compile/run, plugin path via argv or env
- [x] 1.3 Validate: libtpu.so full pass (API v0.69, 4 devices); xla_cuda_plugin.so (jax-cuda12-pjrt wheel) loads, negotiates, fails at Client_Create with the plugin's own "No visible GPU devices" — correct on a GPU-less VM
- [x] 1.4 Full TPU regression after loader changes (gradcheck / dp / gather spot-checked clean)

## 2. Gather embedding

- [x] 2.1 `Op::Gather` + `Op::ScatterAdd`: builders, StableHLO emission (gather + scatter-with-add-region), VJP rules (gather ← scatter-add; scatter-add ← identity + gather)
- [x] 2.2 `nn::embedding` switched to gather; one-hot kept as `embedding_onehot`
- [x] 2.3 `cpp_gather` test: forward exact, gradient exact incl. duplicate-row accumulation (max err 0.00e+00 both)
- [x] 2.4 Regression + 98M bench: loss trajectory identical (9.5171 → 5.7298 @50 steps), 35 ms/step, MFU 25.2%

## 3. Scale

- [x] 3.1 1B replicated feasibility run — **1.0415B params trains on 4 chips**: Bpr=2 → 156 ms/step, MFU 14.9%; Bpr=4 → 188 ms/step, 43.5k tok/s, **MFU 24.7%** (parity with the 98M config); 4.2 GB ckpt roundtrip exact both times
- [x] 3.2 Not needed: 1B fits replicated (GSPMD remains the >2B path, `gpt-scale-benchmark` 6.2)

## 4. Other backends (blocked on resources)

- [ ] 4.1 Build `pjrt_c_api_cpu_plugin.so` from openxla/xla and run the full regression suite on CPU — **blocked: needs 20–50 GB disk, VM has ~5 GB free**
- [ ] 4.2 GPU end-to-end on a CUDA machine via `PJRT_PLUGIN_PATH=…/xla_cuda_plugin.so` — **blocked: no GPU on this VM**
