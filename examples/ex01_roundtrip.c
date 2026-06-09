/*
 * ex01_roundtrip.c — Device enumeration + host↔TPU roundtrip.
 *
 * Compile: make ex01_roundtrip
 * Run:     ./ex01_roundtrip
 */

#include "tpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    printf("=== ex01: Device enumeration + roundtrip ===\n\n");

    tpu_ctx_t* ctx = tpu_init(NULL);
    if (!ctx) {
        fprintf(stderr, "tpu_init failed: %s\n", tpu_strerror(NULL));
        return 1;
    }

    /* Print topology */
    char* topo = tpu_topology_string(ctx);
    if (topo) { printf("%s\n\n", topo); free(topo); }

    /* List all addressable devices */
    int n = tpu_num_addressable_devices(ctx);
    printf("%d addressable device(s):\n", n);
    for (int i = 0; i < n; i++) {
        tpu_device_info_t info;
        if (tpu_device_info(ctx, i, &info)) {
            fprintf(stderr, "device_info(%d): %s\n", i, tpu_strerror(ctx));
            continue;
        }
        printf("  [%d] id=%-3d hw=%-3d kind=%-10s addressable=%s\n",
               i, info.id, info.local_hw_id, info.kind,
               info.is_addressable ? "yes" : "no");
        printf("       %s\n", info.description);
        if (info.bytes_limit_valid)
            printf("       HBM: %lld MB in use / %lld MB total\n",
                   (long long)(info.bytes_in_use >> 20),
                   (long long)(info.bytes_limit  >> 20));
    }
    printf("\n");

    /* Roundtrip: upload float32[2,4] → TPU → download and verify */
    float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    int64_t dims[] = {2, 4};

    printf("Uploading float32[2,4] = {1..8} to device 0...\n");
    tpu_buf_t* buf = tpu_upload_f32(ctx, 0, data, dims, 2);
    if (!buf) { fprintf(stderr, "upload failed: %s\n", tpu_strerror(ctx)); return 1; }

    printf("  On-device size: %zu bytes\n", tpu_buf_size(buf));
    int64_t rdims[8]; size_t rndims;
    tpu_buf_dims(buf, rdims, &rndims);
    printf("  Shape: [");
    for (size_t i = 0; i < rndims; i++) printf("%lld%s", (long long)rdims[i], i+1<rndims?",":"");
    printf("]  dtype: %s\n", tpu_dtype_name(tpu_buf_dtype(buf)));

    printf("Downloading back to host...\n");
    float result[8] = {0};
    if (tpu_download(buf, result, sizeof(result))) {
        fprintf(stderr, "download failed: %s\n", tpu_strerror(ctx)); return 1;
    }

    int ok = 1;
    for (int i = 0; i < 8; i++) if (result[i] != data[i]) ok = 0;
    printf("  Got: [");
    for (int i = 0; i < 8; i++) printf("%.1f%s", result[i], i<7?", ":"");
    printf("]\n");
    printf("  Roundtrip: %s\n\n", ok ? "PASSED" : "FAILED");

    tpu_buf_free(buf);
    tpu_destroy(ctx);
    return ok ? 0 : 1;
}
