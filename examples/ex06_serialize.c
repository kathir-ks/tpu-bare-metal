/*
 * ex06_serialize.c — Compile once, save to disk, load back and rerun.
 *
 * Demonstrates that a compiled TPU executable can be saved and reloaded
 * without recompiling. The serialized format is TPU-version-specific.
 *
 * Requires: matmul.hlo.pb, compile_opts.pb
 *   export TPU_DATA_DIR=<dir>
 *
 * Compile: make ex06_serialize
 * Run:     TPU_DATA_DIR=. ./ex06_serialize
 */

#include "tpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EXEC_FILE "/tmp/tpu_matmul.exec"

static void make_path(char* buf, size_t bufsz, const char* f) {
    const char* d = getenv("TPU_DATA_DIR"); if (!d) d = ".";
    snprintf(buf, bufsz, "%s/%s", d, f);
}

static void run_matmul(tpu_exec_t* exec, tpu_ctx_t* ctx,
                       const float* a, const float* b, float* out,
                       const char* label)
{
    int64_t shape[] = {4, 4};
    tpu_buf_t* a_buf = tpu_upload_f32(ctx, 0, a, shape, 2);
    tpu_buf_t* b_buf = tpu_upload_f32(ctx, 0, b, shape, 2);
    tpu_buf_t* r_buf = tpu_run2(exec, a_buf, b_buf);
    tpu_download(r_buf, out, 16 * sizeof(float));
    printf("[%s] result[0,0] = %.1f  result[3,3] = %.1f\n",
           label, out[0], out[15]);
    tpu_buf_free(r_buf);
    tpu_buf_free(a_buf);
    tpu_buf_free(b_buf);
}

int main(void)
{
    printf("=== ex06: Compile, save, load, rerun ===\n\n");

    tpu_ctx_t* ctx = tpu_init(NULL);
    if (!ctx) { fprintf(stderr, "tpu_init: %s\n", tpu_strerror(NULL)); return 1; }

    float a[16] = {0};
    a[0] = a[5] = a[10] = a[15] = 1.f;  /* identity */
    float b[16];
    for (int i = 0; i < 16; i++) b[i] = (float)(i + 1);

    /* ── Phase 1: compile and save ────────────────────────────────────── */
    printf("Phase 1: compile matmul.hlo.pb...\n");
    char mm_pb[512], opts_pb[512];
    make_path(mm_pb,   sizeof(mm_pb),   "matmul.hlo.pb");
    make_path(opts_pb, sizeof(opts_pb), "compile_opts.pb");
    tpu_exec_t* exec1 = tpu_compile_file(ctx, mm_pb, opts_pb);
    if (!exec1) { fprintf(stderr, "compile: %s\n", tpu_strerror(ctx)); return 1; }

    float r1[16] = {0};
    run_matmul(exec1, ctx, a, b, r1, "original");

    printf("Saving to %s...\n", EXEC_FILE);
    if (tpu_exec_save(exec1, EXEC_FILE)) {
        fprintf(stderr, "save: %s\n", tpu_strerror(ctx)); return 1;
    }
    printf("Saved.\n\n");
    tpu_exec_free(exec1);

    /* ── Phase 2: load from disk and rerun ────────────────────────────── */
    printf("Phase 2: load from %s...\n", EXEC_FILE);
    tpu_exec_t* exec2 = tpu_exec_load(ctx, EXEC_FILE);
    if (!exec2) { fprintf(stderr, "load: %s\n", tpu_strerror(ctx)); return 1; }
    printf("Loaded. num_outputs=%d\n", tpu_exec_num_outputs(exec2));

    float r2[16] = {0};
    run_matmul(exec2, ctx, a, b, r2, "loaded  ");

    int ok = 1;
    for (int i = 0; i < 16; i++) if (fabsf(r1[i] - r2[i]) > 0.001f) ok = 0;
    printf("\nOriginal vs loaded results match: %s\n", ok ? "PASSED" : "FAILED");

    tpu_exec_free(exec2);
    tpu_destroy(ctx);
    return ok ? 0 : 1;
}
