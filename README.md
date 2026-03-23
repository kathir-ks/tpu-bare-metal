# tpu-bare-metal

Direct TPU control via `libtpu.so` PJRT C API — no JAX or Python needed at runtime.

## What This Does

Loads Google's `libtpu.so` shared library at runtime using `dlopen`, accesses the **PJRT C API** function pointer table, and directly:

- Initializes TPU hardware
- Enumerates TPU devices (type, coordinates, memory)
- Transfers data between host RAM and TPU HBM
- Compiles HLO programs (XLA's IR) to TPU executables
- Executes computations on TPU cores

All from pure C. No Python interpreter, no JAX, no TensorFlow.

## Hardware Tested

- **TPU v4** (4 cores, 2×2 topology)
- PJRT API v0.69, TFRT TPU v4 runtime

## Files

| File | Description |
|------|-------------|
| `tpu_pjrt_test.c` | Device enumeration + host↔TPU data transfer roundtrip |
| `tpu_compute.c` | Full pipeline: compile HLO → execute on TPU (element-wise add + matrix multiply) |
| `matmul.mlir` | StableHLO text for matrix multiply (human-readable reference) |
| `gen_hlo.py` | Python script to regenerate HLO protobuf files from JAX |

## Build & Run

### Prerequisites

```bash
# Install libtpu (only needed to get the .so file)
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Compile

```bash
gcc -o tpu_pjrt_test tpu_pjrt_test.c -ldl
gcc -o tpu_compute tpu_compute.c -ldl -lm
```

### Generate HLO programs (one-time)

```bash
python3 gen_hlo.py
```

### Run

```bash
# Device enumeration + data transfer test
./tpu_pjrt_test

# Compile & execute HLO on TPU
./tpu_compute
```

## Example Output

```
[4] Executing x + x on TPU...
    Execution complete!
    Result (x + x):
    [2, 4, 6, 8, 10, 12, 14, 16,
     18, 20, 22, 24, 26, 28, 30, 32]
    Verify: PASSED

[7] Executing A @ B on TPU...
    Result (I @ B = B):
    [   1.0   2.0   3.0   4.0  ]
    [   5.0   6.0   7.0   8.0  ]
    [   9.0  10.0  11.0  12.0  ]
    [  13.0  14.0  15.0  16.0  ]
    Verify (I @ B == B): PASSED

[8] Executing B @ B on TPU...
    Result (B @ B):
    [     90    100    110    120  ]
    [    202    228    254    280  ]
    [    314    356    398    440  ]
    [    426    484    542    600  ]
    Verify: PASSED
```

## How It Works

The PJRT (Pretty Jit RunTime) C API is a hardware-agnostic plugin interface. `libtpu.so` exports `GetPjrtApi()` which returns a struct of ~113 function pointers:

```c
void* handle = dlopen("libtpu.so", RTLD_NOW | RTLD_GLOBAL);
const PJRT_Api* api = ((GetPjrtApiFn)dlsym(handle, "GetPjrtApi"))();

// Header layout: struct_size(8) + extension_start(8) + PJRT_Api_Version(24) = 40 bytes
// Function pointers start at offset 40
void** fn = (void**)((char*)api + 40);

// fn[3]  = PJRT_Plugin_Initialize
// fn[10] = PJRT_Client_Create
// fn[15] = PJRT_Client_Devices
// fn[20] = PJRT_Client_Compile
// fn[22] = PJRT_Client_BufferFromHostBuffer
// fn[55] = PJRT_LoadedExecutable_Execute
// fn[70] = PJRT_Buffer_ToHostBuffer
```

### Key Discovery: PJRT_Api Header Layout

The `PJRT_Api` struct header is **40 bytes** (not the 24 you'd expect):

```
+0:   size_t struct_size          (e.g. 944)
+8:   PJRT_Extension_Base* ext
+16:  PJRT_Api_Version {
        size_t struct_size (=24)
        void*  ext
        int    major (=0)
        int    minor (=69)
      }                           // 24 bytes
+40:  function pointers begin...  // 113 pointers
```

This is because `PJRT_Api_Version` follows the same `{struct_size, extension_start, ...}` pattern as all PJRT structs.

## API Surface of libtpu.so

`libtpu.so` exports 226 symbols across two API layers:

| API | Entry Point | Functions | Status |
|-----|-------------|-----------|--------|
| **PJRT C API** (modern) | `GetPjrtApi()` | 113 | Recommended, used here |
| **Legacy TF/TPU C API** | `Tpu*` symbols | ~100 | Requires TF-style init |
| **SDK API** | `GetLibtpuSdkApi()` | ? | Undocumented |

## References

- [PJRT C API Header (pjrt_c_api.h)](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h)
- [PJRT TPU Entry Point (pjrt_c_api_tpu.h)](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_tpu.h)
- [Legacy TPU Executor C API](https://github.com/openxla/xla/blob/main/xla/stream_executor/tpu/tpu_executor_c_api.h)
- [PJRT Integration Guide](https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md)
