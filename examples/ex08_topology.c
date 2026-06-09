/*
 * ex08_topology.c — Topology string and HBM memory stats per device.
 *
 * Compile: make ex08_topology
 * Run:     ./ex08_topology
 */

#include "tpu.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    printf("=== ex08: Topology + HBM stats ===\n\n");

    tpu_ctx_t* ctx = tpu_init(NULL);
    if (!ctx) { fprintf(stderr, "tpu_init: %s\n", tpu_strerror(NULL)); return 1; }

    /* Print platform topology */
    char* topo = tpu_topology_string(ctx);
    if (topo) {
        printf("─── Platform ───────────────────────────────\n");
        printf("%s\n\n", topo);
        free(topo);
    }

    /* Per-device details */
    int n = tpu_num_addressable_devices(ctx);
    printf("─── Addressable Devices (%d) ─────────────────\n", n);
    for (int i = 0; i < n; i++) {
        tpu_device_info_t info;
        if (tpu_device_info(ctx, i, &info)) {
            printf("  [%d] ERROR: %s\n", i, tpu_strerror(ctx));
            continue;
        }
        printf("  [%d] id=%-4d hw_id=%-4d kind=%-12s addressable=%s\n",
               i, info.id, info.local_hw_id, info.kind,
               info.is_addressable ? "yes" : "no");
        printf("       %s\n", info.description);
        if (info.bytes_limit_valid) {
            double hbm_gb  = (double)info.bytes_limit  / (1024.0 * 1024 * 1024);
            double used_mb = (double)info.bytes_in_use / (1024.0 * 1024);
            printf("       HBM: %.2f GB total  %.1f MB in use\n", hbm_gb, used_mb);
        } else {
            printf("       HBM: stats unavailable\n");
        }
    }

    printf("\nTotal devices (including non-addressable): %d\n",
           tpu_num_devices(ctx));

    /* Print dtype sizes for reference */
    printf("\n─── Dtype reference ─────────────────────────\n");
    tpu_dtype_t types[] = {
        TPU_DTYPE_F32, TPU_DTYPE_F64, TPU_DTYPE_BF16,
        TPU_DTYPE_F16, TPU_DTYPE_S32, TPU_DTYPE_S8
    };
    for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); i++) {
        printf("  %-6s  %zu bytes\n",
               tpu_dtype_name(types[i]), tpu_dtype_size(types[i]));
    }

    tpu_destroy(ctx);
    return 0;
}
