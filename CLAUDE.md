# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Bare-metal C programs that drive TPU hardware directly via `libtpu.so`'s PJRT C API — no Python, JAX, or TensorFlow at runtime. The library is loaded with `dlopen`; all 113 function pointers are accessed by index into the table at `api_ptr + 40`.

Tested on TPU v4 (4 cores, 2x2 topology), PJRT API v0.69.

## Build

```bash
make          # builds originals + framework library + all 8 examples
make lib      # framework/libtpu_fw.a only
make examples # all 8 example programs
make hlo      # regenerate *.pb HLO files (needs JAX venv, see below)
make clean
```

The original demos still compile individually:
```bash
gcc -Wall -O2 -o tpu_pjrt_test tpu_pjrt_test.c -ldl
gcc -Wall -O2 -o tpu_compute   tpu_compute.c   -ldl -lm
```

## Run

```bash
# Original proofs-of-concept (standalone, no framework)
./tpu_pjrt_test
./tpu_compute

# Framework examples (require HLO files; set TPU_DATA_DIR first)
export TPU_DATA_DIR=.
./ex01_roundtrip           # device list + host<->TPU roundtrip
./ex02_matmul              # compile HLO + execute add and matmul
./ex03_spmd                # same kernel across all 4 devices simultaneously
./ex04_dev2dev             # device-to-device copy (no host roundtrip)
./ex05_async               # async execution with OnReady callback
./ex06_serialize           # compile -> save to disk -> load -> rerun
./ex07_shard               # split matrix across N devices
./ex08_topology            # topology string + HBM capacity per device
```

The `libtpu.so` path defaults to `/home/kathirks_gc/.local/lib/python3.10/site-packages/libtpu/libtpu.so`. Override with `LIBTPU_PATH=/path/to/libtpu.so`.

## Prerequisite: .pb files for HLO examples

Both the legacy `tpu_compute` demo and the framework examples need generated `.pb` files:

```bash
source ~/venv-maxtext-py312/bin/activate
export TPU_DATA_DIR=.   # framework examples read from here
python3 gen_hlo.py      # writes add.hlo.pb, matmul.hlo.pb, compile_opts.pb
```

The legacy `tpu_compute` reads from the **hardcoded path `/home/kathirks_gc/tpu_direct/`**. To use it:
```bash
export TPU_DATA_DIR=~/tpu_direct && python3 gen_hlo.py
```

## Framework (`framework/` + `examples/`)

### Quick start

```c
#include "tpu.h"

tpu_ctx_t*  ctx  = tpu_init(NULL);   // NULL -> LIBTPU_PATH env or default
tpu_exec_t* exec = tpu_compile_file(ctx, "matmul.hlo.pb", NULL);  // NULL opts = defaults
int64_t dims[] = {4, 4};
tpu_buf_t*  a   = tpu_upload_f32(ctx, 0, data_a, dims, 2);
tpu_buf_t*  b   = tpu_upload_f32(ctx, 0, data_b, dims, 2);
tpu_buf_t*  out = tpu_run2(exec, a, b);
tpu_download(out, result, sizeof(result));
tpu_buf_free(a); tpu_buf_free(b); tpu_buf_free(out);
tpu_exec_free(exec);
tpu_destroy(ctx);
```

### New capabilities vs the original demos

| Feature | API | PJRT fn index |
|---------|-----|---------------|
| Multi-device SPMD | `tpu_run_spmd` | fn[55] num_devices > 1 |
| Buffer sharding | `tpu_shard_f32` | upload per device |
| Device-to-device copy | `tpu_buf_copy_to_device` | fn[69] |
| Async execution | `tpu_run_async` | fn[9] OnReady callback |
| Executable save/load | `tpu_exec_save` / `tpu_exec_load` | fn[49] / fn[56] |
| Topology / HBM stats | `tpu_topology_string`, `tpu_device_info` | fn[34] MemoryStats |
| Recoverable errors | `tpu_strerror(ctx)` | no more `exit(1)` |

### File layout

```
framework/
  tpu_pjrt.h     -- internal: all 113 fn indices, PJRT arg structs
  tpu.h          -- public API (no PJRT types exposed)
  tpu.c          -- core: init, upload, download, compile, run, async, dev2dev
  tpu_spmd.c     -- tpu_run_spmd, tpu_shard_f32
  tpu_serial.c   -- tpu_exec_save, tpu_exec_load

examples/
  ex01_roundtrip.c   ex02_matmul.c    ex03_spmd.c      ex04_dev2dev.c
  ex05_async.c       ex06_serialize.c ex07_shard.c     ex08_topology.c
```

### Error handling

All framework functions return NULL or -1 on error:
```c
const char* msg = tpu_strerror(ctx);   // NULL if no error
```

## Architecture

### PJRT_Api function table layout

The `GetPjrtApi()` return value is a struct whose first 40 bytes are a header:

```
+0:   size_t struct_size
+8:   PJRT_Extension_Base* extension_start
+16:  PJRT_Api_Version { struct_size(8), ext(8), major(4), minor(4) }  // 24 bytes
+40:  function pointers [0..112]
```

`framework/tpu_pjrt.h` defines `PJRT_CALL(fn_table, idx, T, a)` and all 113 indices.
The original demos use local `CALL`/`CALLV` macros with a global `fn` pointer; the
framework passes `fn_table` explicitly to support multiple contexts.

### PJRT arg struct convention

Every PJRT arg struct starts with `{size_t struct_size; void* extension_start}`. Always
set `struct_size = sizeof(args)` before calling. Output fields are written back into the
same struct by PJRT.

### Key function indices

| Index | Function |
|-------|----------|
| 0/1 | Error destroy/message |
| 3 | Plugin_Initialize |
| 5/8/9 | Event destroy/await/OnReady |
| 10/11 | Client create/destroy |
| 15/16 | Client_Devices / AddressableDevices |
| 20 | Client_Compile |
| 22 | Client_BufferFromHostBuffer |
| 49/56 | Executable serialize/deserialize+load |
| 50/55 | LoadedExecutable destroy/execute |
| 58/69/70 | Buffer destroy/CopyToDevice/ToHostBuffer |

### Original file roles (unchanged)

- **`tpu_pjrt_test.c`** — standalone demo: init, enumerate devices, host->TPU->host roundtrip.
- **`tpu_compute.c`** — full pipeline: load HLO protobufs, compile, execute, verify.
- **`gen_hlo.py`** — uses JAX to generate HLO protobufs and `CompileOptions`. Requires `~/venv-maxtext-py312`. Respects `$TPU_DATA_DIR` for output path.
- **`matmul.mlir`** — human-readable StableHLO (generated by `gen_hlo.py`), not used at runtime.

### Adding a new computation

1. Add a JAX function to `gen_hlo.py` and serialize its HLO
2. In C: `tpu_compile_file(ctx, "my_op.hlo.pb", NULL)`
3. Upload inputs with `tpu_upload_f32`, run with `tpu_run`, download with `tpu_download`
