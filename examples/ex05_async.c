/*
 * ex05_async.c — Async execution with PJRT_Event_OnReady callback.
 *
 * Launches TPU execution and returns immediately. A callback fires when
 * the result is ready. Main thread polls a volatile flag (demo use only;
 * production code should use a mutex + condvar or similar).
 *
 * Requires: add.hlo.pb, compile_opts.pb
 *   export TPU_DATA_DIR=<dir>
 *
 * Compile: make ex05_async
 * Run:     TPU_DATA_DIR=. ./ex05_async
 */

#include "tpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── Shared state between main and callback ────────────────────────────── */

typedef struct {
    volatile int done;
    char         error[256];
    float        result[16];
    tpu_buf_t*   out_buf;   /* set before launch, read in callback */
} async_state_t;

static void on_ready(const char* error_msg, void* user_arg)
{
    async_state_t* s = (async_state_t*)user_arg;
    if (error_msg) {
        strncpy(s->error, error_msg, sizeof(s->error) - 1);
    } else {
        /* Safe to download: execution is complete */
        tpu_download(s->out_buf, s->result, sizeof(s->result));
        s->error[0] = '\0';
    }
    s->done = 1;  /* signal main thread */
}

static void make_path(char* buf, size_t bufsz, const char* f) {
    const char* d = getenv("TPU_DATA_DIR"); if (!d) d = ".";
    snprintf(buf, bufsz, "%s/%s", d, f);
}

int main(void)
{
    printf("=== ex05: Async execution with OnReady callback ===\n\n");

    tpu_ctx_t* ctx = tpu_init(NULL);
    if (!ctx) { fprintf(stderr, "tpu_init: %s\n", tpu_strerror(NULL)); return 1; }

    char add_pb[512], opts_pb[512];
    make_path(add_pb,  sizeof(add_pb),  "add.hlo.pb");
    make_path(opts_pb, sizeof(opts_pb), "compile_opts.pb");
    tpu_exec_t* exec = tpu_compile_file(ctx, add_pb, opts_pb);
    if (!exec) { fprintf(stderr, "compile: %s\n", tpu_strerror(ctx)); return 1; }

    /* Upload input */
    float x[16];
    for (int i = 0; i < 16; i++) x[i] = (float)(i + 1);
    int64_t shape[] = {4, 4};
    tpu_buf_t* x_buf = tpu_upload_f32(ctx, 0, x, shape, 2);
    if (!x_buf) { fprintf(stderr, "upload: %s\n", tpu_strerror(ctx)); return 1; }

    /* Allocate output buffer handle (framework populates it during run_async) */
    tpu_buf_t* out_buf = NULL;

    async_state_t state;
    memset(&state, 0, sizeof(state));
    state.done    = 0;
    state.out_buf = NULL;

    /* Launch async — returns immediately */
    printf("Launching async execution...\n");
    tpu_buf_t* const args[] = { x_buf };
    if (tpu_run_async(exec, 0, args, 1, &out_buf, 1, on_ready, &state)) {
        fprintf(stderr, "run_async: %s\n", tpu_strerror(ctx)); return 1;
    }
    state.out_buf = out_buf;
    printf("Launch returned. Waiting for callback...\n");

    /* Poll until done (real code: use pthread_cond_wait or similar) */
    struct timespec ts = {0, 1000000};  /* 1ms */
    int iters = 0;
    while (!state.done) {
        nanosleep(&ts, NULL);
        iters++;
    }
    printf("Callback fired after ~%d ms\n", iters);

    if (state.error[0]) {
        fprintf(stderr, "Async error: %s\n", state.error);
        return 1;
    }

    /* Verify x + x */
    int ok = 1;
    for (int i = 0; i < 16; i++)
        if (fabsf(state.result[i] - x[i] * 2.f) > 0.001f) ok = 0;

    printf("Result (x + x): [%.0f, %.0f, ..., %.0f]\n",
           state.result[0], state.result[1], state.result[15]);
    printf("Async execution: %s\n\n", ok ? "PASSED" : "FAILED");

    tpu_buf_free(out_buf);
    tpu_buf_free(x_buf);
    tpu_exec_free(exec);
    tpu_destroy(ctx);
    return ok ? 0 : 1;
}
