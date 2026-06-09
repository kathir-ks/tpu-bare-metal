/*
 * ex04_dev2dev.c — Device-to-device buffer copy (no host roundtrip).
 *
 * Uploads a buffer to device 0, copies it directly to device 1 using
 * PJRT_Buffer_CopyToDevice, then downloads from device 1 and verifies.
 *
 * Requires at least 2 addressable devices.
 *
 * Compile: make ex04_dev2dev
 * Run:     ./ex04_dev2dev
 */

#include "tpu.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    printf("=== ex04: Device-to-device copy ===\n\n");

    tpu_ctx_t* ctx = tpu_init(NULL);
    if (!ctx) { fprintf(stderr, "tpu_init: %s\n", tpu_strerror(NULL)); return 1; }

    int ndevs = tpu_num_addressable_devices(ctx);
    printf("Addressable devices: %d\n", ndevs);
    if (ndevs < 2) {
        printf("Need at least 2 devices; skipping dev2dev test.\n\n");
        /* Still verify single-device roundtrip */
        float data[8];
        for (int i = 0; i < 8; i++) data[i] = (float)(i + 1);
        int64_t dims[] = {8};
        tpu_buf_t* b = tpu_upload_f32(ctx, 0, data, dims, 1);
        float out[8] = {0};
        tpu_download(b, out, sizeof(out));
        printf("Single-device roundtrip: ");
        int ok = 1;
        for (int i = 0; i < 8; i++) if (out[i] != data[i]) ok = 0;
        printf("%s\n", ok ? "PASSED" : "FAILED");
        tpu_buf_free(b);
        tpu_destroy(ctx);
        return 0;
    }

    /* Upload data to device 0 */
    float data[16];
    for (int i = 0; i < 16; i++) data[i] = (float)(i + 1);
    int64_t dims[] = {4, 4};

    printf("Uploading float32[4,4] to device 0...\n");
    tpu_buf_t* src = tpu_upload_f32(ctx, 0, data, dims, 2);
    if (!src) { fprintf(stderr, "upload: %s\n", tpu_strerror(ctx)); return 1; }
    printf("  Device-0 buffer: size=%zu bytes\n", tpu_buf_size(src));

    /* Copy device 0 → device 1 (no host memory involved) */
    printf("Copying device 0 → device 1 (no host roundtrip)...\n");
    tpu_buf_t* dst = tpu_buf_copy_to_device(src, 1);
    if (!dst) { fprintf(stderr, "copy_to_device: %s\n", tpu_strerror(ctx)); return 1; }
    printf("  Device-1 buffer: size=%zu bytes\n", tpu_buf_size(dst));

    /* Download from device 1 and verify */
    printf("Downloading from device 1...\n");
    float result[16] = {0};
    if (tpu_download(dst, result, sizeof(result))) {
        fprintf(stderr, "download: %s\n", tpu_strerror(ctx)); return 1;
    }

    int ok = 1;
    for (int i = 0; i < 16; i++) if (result[i] != data[i]) ok = 0;

    printf("  Got: [%.0f, %.0f, ..., %.0f]\n", result[0], result[1], result[15]);
    printf("Device-to-device copy: %s\n\n", ok ? "PASSED" : "FAILED");

    tpu_buf_free(src);
    tpu_buf_free(dst);
    tpu_destroy(ctx);
    return ok ? 0 : 1;
}
