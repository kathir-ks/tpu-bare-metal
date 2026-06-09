/*
 * tpu.h — Public API for the TPU PJRT C framework.
 *
 * Pure C99. No PJRT types leak through this header.
 * Link with framework/libtpu_fw.a and -ldl -lm.
 *
 * Quick start:
 *   tpu_ctx_t*  ctx  = tpu_init(NULL);
 *   tpu_exec_t* exec = tpu_compile_file(ctx, "matmul.hlo.pb", NULL);
 *   int64_t dims[] = {4, 4};
 *   tpu_buf_t*  a   = tpu_upload_f32(ctx, 0, data_a, dims, 2);
 *   tpu_buf_t*  b   = tpu_upload_f32(ctx, 0, data_b, dims, 2);
 *   tpu_buf_t*  out = tpu_run2(exec, a, b);
 *   tpu_download(out, result, sizeof(result));
 *   tpu_buf_free(a); tpu_buf_free(b); tpu_buf_free(out);
 *   tpu_exec_free(exec);
 *   tpu_destroy(ctx);
 */

#ifndef TPU_H
#define TPU_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque framework types ───────────────────────────────────────────────── */

typedef struct tpu_ctx_t    tpu_ctx_t;   /* one per process */
typedef struct tpu_buf_t    tpu_buf_t;   /* device buffer handle */
typedef struct tpu_exec_t   tpu_exec_t;  /* compiled executable handle */

/* ── Element type enum ────────────────────────────────────────────────────── */
/* Values match PJRT_Buffer_Type directly; zero-overhead cast. */

typedef enum {
    TPU_DTYPE_INVALID = 0,
    TPU_DTYPE_PRED    = 1,
    TPU_DTYPE_S8      = 2,
    TPU_DTYPE_S16     = 3,
    TPU_DTYPE_S32     = 4,
    TPU_DTYPE_S64     = 5,
    TPU_DTYPE_U8      = 6,
    TPU_DTYPE_U16     = 7,
    TPU_DTYPE_U32     = 8,
    TPU_DTYPE_U64     = 9,
    TPU_DTYPE_F16     = 10,
    TPU_DTYPE_F32     = 11,
    TPU_DTYPE_F64     = 12,
    TPU_DTYPE_BF16    = 13,
    TPU_DTYPE_C64     = 14,
    TPU_DTYPE_C128    = 15,
} tpu_dtype_t;

/* Element size in bytes; returns 0 for unknown types. */
size_t tpu_dtype_size(tpu_dtype_t dtype);
/* Printable name, e.g. "f32". */
const char* tpu_dtype_name(tpu_dtype_t dtype);

/* ── Device info ──────────────────────────────────────────────────────────── */

typedef struct {
    int     id;
    int     local_hw_id;
    bool    is_addressable;
    char    kind[64];         /* e.g. "TPU v4" */
    char    description[256]; /* full human-readable string */
    int64_t bytes_in_use;
    int64_t bytes_limit;      /* HBM capacity; valid only if bytes_limit_valid */
    bool    bytes_limit_valid;
} tpu_device_info_t;

/* ── Async callback ───────────────────────────────────────────────────────── */
/* Called on a PJRT-internal thread. Never call tpu_* functions from it.
 * error_msg is NULL on success; a NUL-terminated string on failure.
 * user_arg is the value passed to tpu_run_async. */

typedef void (*tpu_on_ready_fn)(const char* error_msg, void* user_arg);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Init / teardown                                                             */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Load libtpu.so, call Plugin_Initialize, create client, cache device list.
 * lib_path: path to libtpu.so; NULL → LIBTPU_PATH env var → built-in default.
 * Returns NULL on failure; call tpu_strerror(NULL) for the message. */
tpu_ctx_t*  tpu_init(const char* lib_path);

/* Destroy the client and dlclose the library. Frees ctx. */
void        tpu_destroy(tpu_ctx_t* ctx);

/* Last error message for ctx (pass NULL for the global init error).
 * Valid until the next tpu_* call on the same ctx. NULL means no error. */
const char* tpu_strerror(tpu_ctx_t* ctx);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Device queries                                                              */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Total devices visible to client (including non-addressable in multi-process). */
int   tpu_num_devices(tpu_ctx_t* ctx);

/* Addressable devices — the useful count for single-process programs. */
int   tpu_num_addressable_devices(tpu_ctx_t* ctx);

/* Fill info for addressable device dev_idx (0-based).
 * Returns 0 on success, -1 on error. */
int   tpu_device_info(tpu_ctx_t* ctx, int dev_idx, tpu_device_info_t* out);

/* Returns a malloc'd human-readable topology/platform string, or NULL on error.
 * Caller must free() the result. */
char* tpu_topology_string(tpu_ctx_t* ctx);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Compile                                                                     */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Compile from a binary HLO protobuf file.
 * opts_path: path to a serialized CompileOptions protobuf, or NULL for defaults
 *   (num_replicas=1, num_partitions=1).
 * Returns NULL on error. */
tpu_exec_t* tpu_compile_file(tpu_ctx_t* ctx,
                              const char* hlo_pb_path,
                              const char* opts_pb_path);

/* Compile from an in-memory buffer.
 * format: "hlo" for binary HLO protobuf, "mlir" for StableHLO text/bytecode.
 * opts / opts_sz: serialized CompileOptions, or (NULL, 0) for defaults.
 * Returns NULL on error. */
tpu_exec_t* tpu_compile_buf(tpu_ctx_t* ctx,
                             const void* code, size_t code_sz,
                             const char* format,
                             const void* opts, size_t opts_sz);

/* Number of outputs per device. Returns -1 on error. */
int  tpu_exec_num_outputs(tpu_exec_t* exec);

/* Addressable device indices for this executable, in replica order.
 * out_dev_indices[r] = addressable device index for replica r.
 * Returns 0 on success, -1 on error. */
int  tpu_exec_device_order(tpu_exec_t* exec, int* out_dev_indices, size_t n);

/* Free a compiled executable. */
void tpu_exec_free(tpu_exec_t* exec);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Executable serialization  (tpu_serial.c)                                   */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Serialize a compiled executable to a file.
 * Returns 0 on success, -1 on error.
 * Note: serialized format is TPU-version-specific and not portable. */
int  tpu_exec_save(tpu_exec_t* exec, const char* path);

/* Load a previously saved executable.
 * Returns NULL on error. */
tpu_exec_t* tpu_exec_load(tpu_ctx_t* ctx, const char* path);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Buffer upload  (host → TPU HBM)                                            */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Generic typed upload. data must have the correct byte count for dtype+dims.
 * dev_idx: 0-based addressable device index.
 * Returns NULL on error. */
tpu_buf_t* tpu_upload(tpu_ctx_t* ctx, int dev_idx,
                      const void* data, tpu_dtype_t dtype,
                      const int64_t* dims, size_t ndims);

/* Convenience wrappers for common dtypes. */
tpu_buf_t* tpu_upload_f32(tpu_ctx_t* ctx, int dev_idx,
                           const float* data,
                           const int64_t* dims, size_t ndims);

tpu_buf_t* tpu_upload_bf16(tpu_ctx_t* ctx, int dev_idx,
                            const void* data,  /* uint16_t[] holding BF16 bits */
                            const int64_t* dims, size_t ndims);

tpu_buf_t* tpu_upload_s32(tpu_ctx_t* ctx, int dev_idx,
                           const int32_t* data,
                           const int64_t* dims, size_t ndims);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Buffer download  (TPU HBM → host)                                          */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Copy buffer contents to pre-allocated host memory.
 * dst_size must be >= tpu_buf_size(buf).
 * Returns 0 on success, -1 on error. */
int    tpu_download(tpu_buf_t* buf, void* dst, size_t dst_size);

/* On-device size in bytes. Returns 0 on error. */
size_t tpu_buf_size(tpu_buf_t* buf);

/* Fill dims[0..ndims-1] and set *ndims. dims must be at least 8 elements.
 * Returns 0 on success, -1 on error. */
int    tpu_buf_dims(tpu_buf_t* buf, int64_t* dims, size_t* ndims);

/* Element type of the buffer. */
tpu_dtype_t tpu_buf_dtype(tpu_buf_t* buf);

/* Free a buffer (PJRT_Buffer_Destroy). */
void   tpu_buf_free(tpu_buf_t* buf);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Single-device execution                                                     */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Run executable on one device. Blocks until completion.
 * dev_idx: 0-based addressable device index.
 * outputs: caller-allocated array of tpu_buf_t* of length noutputs.
 *   Framework allocates each *outputs[i]; caller must tpu_buf_free them.
 * Returns 0 on success, -1 on error. */
int tpu_run(tpu_exec_t* exec, int dev_idx,
            tpu_buf_t* const* args, size_t nargs,
            tpu_buf_t** outputs, size_t noutputs);

/* Shorthands: run on device 0, return first output (caller frees).
 * Return NULL on error. */
tpu_buf_t* tpu_run1(tpu_exec_t* exec, tpu_buf_t* a);
tpu_buf_t* tpu_run2(tpu_exec_t* exec, tpu_buf_t* a, tpu_buf_t* b);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Async single-device execution                                               */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Launch execution and return immediately.
 * cb(error_msg, user_arg) is called from a PJRT-internal thread when done.
 * outputs[] are valid when cb fires (even if cb reports an error — free them).
 * Caller must keep args alive until cb fires.
 * Returns 0 if launch succeeded (does NOT wait for completion). */
int tpu_run_async(tpu_exec_t* exec, int dev_idx,
                  tpu_buf_t* const* args, size_t nargs,
                  tpu_buf_t** outputs, size_t noutputs,
                  tpu_on_ready_fn cb, void* user_arg);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Multi-device SPMD execution  (tpu_spmd.c)                                  */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Run the same executable across ndevs devices simultaneously.
 * arg_lists[dev][arg_idx] — input buffers per device.
 * *out_lists is allocated by the framework: (*out_lists)[dev][out_idx].
 *   Call tpu_spmd_outputs_free() to release.
 * ndevs must equal tpu_num_addressable_devices(ctx).
 * Returns 0 on success, -1 on error. */
int  tpu_run_spmd(tpu_exec_t* exec,
                  tpu_buf_t* const* const* arg_lists,
                  size_t ndevs, size_t nargs,
                  tpu_buf_t*** out_lists, size_t noutputs);

/* Free the output array produced by tpu_run_spmd. */
void tpu_spmd_outputs_free(tpu_buf_t** out_lists,
                            size_t ndevs, size_t noutputs);

/* Split float32 host data across ndevs devices along the outermost dimension.
 * dims[0] must be divisible by ndevs.
 * shards: caller-allocated array of tpu_buf_t* of length ndevs; framework
 *   allocates each *shards[d] on device d; caller must tpu_buf_free them.
 * Returns 0 on success, -1 on error. */
int  tpu_shard_f32(tpu_ctx_t* ctx,
                   const float* data,
                   const int64_t* dims, size_t ndims,
                   size_t ndevs,
                   tpu_buf_t** shards);

/* ══════════════════════════════════════════════════════════════════════════ */
/* Device-to-device copy                                                       */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Copy buffer from its current device to dst_dev_idx (0-based addressable).
 * Uses PJRT_Buffer_CopyToDevice — no host memory involved.
 * Returns NULL on error. */
tpu_buf_t* tpu_buf_copy_to_device(tpu_buf_t* src, int dst_dev_idx);

#ifdef __cplusplus
}
#endif

#endif /* TPU_H */
