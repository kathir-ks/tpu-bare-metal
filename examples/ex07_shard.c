/*
 * ex07_shard.c — Split a matrix across N devices along the batch dimension.
 *
 * Takes a [N*4, 4] float32 matrix (where N = num_devices),
 * shards it: replica r gets rows [r*4 .. r*4+3],
 * runs x+x on each shard via tpu_run_spmd,
 * downloads each shard result and verifies.
 *
 * Uses compile_opts_n4.pb (num_replicas=4) and tpu_exec_device_order() to
 * upload each shard to the physical device assigned to its replica.
 *
 * Requires: add.hlo.pb, compile_opts_n4.pb
 *   export TPU_DATA_DIR=<dir>
 *
 * Compile: make ex07_shard
 * Run:     TPU_DATA_DIR=. ./ex07_shard
 */

#include "tpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static void make_path(char* buf, size_t bufsz, const char* f) {
    const char* d = getenv("TPU_DATA_DIR"); if (!d) d = ".";
    snprintf(buf, bufsz, "%s/%s", d, f);
}

int main(void)
{
    printf("=== ex07: Buffer sharding across devices ===\n\n");

    tpu_ctx_t* ctx = tpu_init(NULL);
    if (!ctx) { fprintf(stderr, "tpu_init: %s\n", tpu_strerror(NULL)); return 1; }

    int ndevs = tpu_num_addressable_devices(ctx);
    printf("Devices: %d\n", ndevs);

    /* Compile add (x + x) with num_replicas=ndevs */
    char add_pb[512], opts_pb[512];
    make_path(add_pb,  sizeof(add_pb),  "add.hlo.pb");
    make_path(opts_pb, sizeof(opts_pb), "compile_opts_n4.pb");
    tpu_exec_t* exec = tpu_compile_file(ctx, add_pb, opts_pb);
    if (!exec) { fprintf(stderr, "compile: %s\n", tpu_strerror(ctx)); return 1; }

    /* Find replica-to-device assignment */
    int* dev_order = (int*)malloc((size_t)ndevs * sizeof(int));
    if (tpu_exec_device_order(exec, dev_order, (size_t)ndevs)) {
        fprintf(stderr, "device_order: %s\n", tpu_strerror(ctx)); return 1;
    }
    printf("Replica-to-device: ");
    for (int r = 0; r < ndevs; r++) printf("%d→%d ", r, dev_order[r]);
    printf("\n\n");

    /* Build full matrix: [(ndevs*4), 4].
     * Row row contains values {row*4+1, row*4+2, row*4+3, row*4+4} */
    int total_rows = ndevs * 4;
    float* full = (float*)malloc((size_t)(total_rows * 4) * sizeof(float));
    for (int row = 0; row < total_rows; row++)
        for (int c = 0; c < 4; c++)
            full[row * 4 + c] = (float)(row * 4 + c + 1);

    printf("Full matrix: %d x 4\n", total_rows);
    printf("Each replica gets 4 rows.\n\n");

    /* Upload shard r to the physical device assigned to replica r */
    tpu_buf_t** shards = (tpu_buf_t**)malloc((size_t)ndevs * sizeof(tpu_buf_t*));
    int64_t shard_dims[] = {4, 4};
    for (int r = 0; r < ndevs; r++) {
        const float* shard_data = full + r * 4 * 4;
        shards[r] = tpu_upload_f32(ctx, dev_order[r], shard_data, shard_dims, 2);
        if (!shards[r]) { fprintf(stderr, "upload shard %d: %s\n", r, tpu_strerror(ctx)); return 1; }
    }
    printf("Shards uploaded. Running x+x on each device via SPMD...\n\n");

    /* Build per-replica arg lists */
    tpu_buf_t*** arg_lists = (tpu_buf_t***)malloc((size_t)ndevs * sizeof(tpu_buf_t**));
    for (int r = 0; r < ndevs; r++)
        arg_lists[r] = &shards[r];

    tpu_buf_t** out_lists = NULL;
    if (tpu_run_spmd(exec,
                     (tpu_buf_t* const* const*)arg_lists,
                     (size_t)ndevs, 1,
                     &out_lists, 1)) {
        fprintf(stderr, "run_spmd: %s\n", tpu_strerror(ctx));
        free(arg_lists); free(dev_order); return 1;
    }
    free(arg_lists);

    /* Verify: result[r][row][c] == 2 * (global_row * 4 + c + 1) */
    int all_ok = 1;
    for (int r = 0; r < ndevs; r++) {
        float result[16] = {0};
        tpu_download(out_lists[r * 1 + 0], result, sizeof(result));

        int ok = 1;
        for (int row = 0; row < 4; row++)
            for (int c = 0; c < 4; c++) {
                float expected = 2.f * (float)((r * 4 + row) * 4 + c + 1);
                if (fabsf(result[row*4+c] - expected) > 0.001f) ok = 0;
            }

        printf("Replica %d (dev %d): rows [%d..%d]  result[0]=%.0f ... result[15]=%.0f  %s\n",
               r, dev_order[r], r*4, r*4+3, result[0], result[15], ok ? "OK" : "WRONG");
        if (!ok) all_ok = 0;
    }

    printf("\nSharding: %s\n", all_ok ? "PASSED" : "FAILED");

    tpu_spmd_outputs_free(out_lists, (size_t)ndevs, 1);
    for (int r = 0; r < ndevs; r++) tpu_buf_free(shards[r]);
    free(shards); free(full); free(dev_order);
    tpu_exec_free(exec);
    tpu_destroy(ctx);
    return all_ok ? 0 : 1;
}
