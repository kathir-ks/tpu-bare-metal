/*
 * tpu_spmd.c — Multi-device SPMD execution and buffer sharding.
 */

#include "tpu.h"
#include "tpu_pjrt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Access ctx internals — defined in tpu.c */
struct tpu_ctx_t {
    void*         dl_handle;
    void*         api_ptr;
    void**        fn;
    PJRT_Client*  client;
    PJRT_Device** all_devices;
    size_t        num_all;
    PJRT_Device** addr_devices;
    size_t        num_addr;
    char          errmsg[512];
};

struct tpu_buf_t {
    PJRT_Buffer* pjrt_buf;
    tpu_ctx_t*   ctx;
};

struct tpu_exec_t {
    PJRT_LoadedExecutable* lexec;
    tpu_ctx_t*             ctx;
    size_t                 num_outputs;
};

/* Shared error helper (duplicated from tpu.c to avoid a hidden-linkage dependency) */
static int spmd_err(tpu_ctx_t* ctx, PJRT_Error* err, const char* where)
{
    if (!err) return 0;
    PJRT_Error_Message_Args ma;
    memset(&ma, 0, sizeof(ma));
    ma.struct_size = sizeof(ma);
    ma.error       = err;
    PJRT_CALLV(ctx->fn, FN_ERROR_MESSAGE, PJRT_Error_Message_Args, &ma);
    snprintf(ctx->errmsg, sizeof(ctx->errmsg), "[%s] %.*s",
             where, (int)ma.message_size, ma.message);
    PJRT_Error_Destroy_Args da;
    memset(&da, 0, sizeof(da));
    da.struct_size = sizeof(da);
    da.error       = err;
    PJRT_CALLV(ctx->fn, FN_ERROR_DESTROY, PJRT_Error_Destroy_Args, &da);
    return -1;
}

static int spmd_await(tpu_ctx_t* ctx, PJRT_Event* ev, const char* where)
{
    if (!ev) return 0;
    PJRT_Event_Await_Args aw;
    memset(&aw, 0, sizeof(aw));
    aw.struct_size = sizeof(aw);
    aw.event       = ev;
    int rc = spmd_err(ctx, PJRT_CALL(ctx->fn, FN_EVENT_AWAIT,
                                     PJRT_Event_Await_Args, &aw), where);
    PJRT_Event_Destroy_Args ed;
    memset(&ed, 0, sizeof(ed));
    ed.struct_size = sizeof(ed);
    ed.event       = ev;
    PJRT_CALLV(ctx->fn, FN_EVENT_DESTROY, PJRT_Event_Destroy_Args, &ed);
    return rc;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* tpu_run_spmd                                                                */
/* ══════════════════════════════════════════════════════════════════════════ */

int tpu_run_spmd(tpu_exec_t* exec,
                 tpu_buf_t* const* const* arg_lists,
                 size_t ndevs, size_t nargs,
                 tpu_buf_t*** out_lists, size_t noutputs)
{
    tpu_ctx_t* ctx = exec->ctx;
    if (ndevs != ctx->num_addr) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg),
                 "ndevs=%zu but num_addressable_devices=%zu", ndevs, ctx->num_addr);
        return -1;
    }

    PJRT_Buffer*** inner     = NULL;
    PJRT_Buffer*** pjrt_outs = NULL;
    PJRT_Event**   events    = NULL;

    /* Build 2D input array: [ndevs][nargs] PJRT_Buffer* */
    inner = (PJRT_Buffer***)calloc(ndevs, sizeof(PJRT_Buffer**));
    if (!inner) goto oom;
    for (size_t d = 0; d < ndevs; d++) {
        inner[d] = (PJRT_Buffer**)calloc(nargs, sizeof(PJRT_Buffer*));
        if (!inner[d]) goto oom;
        for (size_t a = 0; a < nargs; a++)
            inner[d][a] = arg_lists[d][a]->pjrt_buf;
    }

    /* Build 2D output array: [ndevs][noutputs] PJRT_Buffer* */
    pjrt_outs = (PJRT_Buffer***)calloc(ndevs, sizeof(PJRT_Buffer**));
    if (!pjrt_outs) goto oom;
    for (size_t d = 0; d < ndevs; d++) {
        pjrt_outs[d] = (PJRT_Buffer**)calloc(noutputs, sizeof(PJRT_Buffer*));
        if (!pjrt_outs[d]) goto oom;
    }

    /* Completion events: one per device */
    events = (PJRT_Event**)calloc(ndevs, sizeof(PJRT_Event*));
    if (!events) goto oom;

    PJRT_ExecuteOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = sizeof(opts);

    PJRT_LoadedExecutable_Execute_Args ex;
    memset(&ex, 0, sizeof(ex));
    ex.struct_size            = sizeof(ex);
    ex.executable             = exec->lexec;
    ex.options                = &opts;
    ex.argument_lists         = (PJRT_Buffer* const* const*)inner;
    ex.num_devices            = ndevs;
    ex.num_args               = nargs;
    ex.output_lists           = (PJRT_Buffer** const*)pjrt_outs;
    ex.device_complete_events = events;
    ex.execute_device         = NULL;   /* all addressable devices */

    if (spmd_err(ctx, PJRT_CALL(ctx->fn, FN_LEXEC_EXECUTE,
                                PJRT_LoadedExecutable_Execute_Args, &ex), "SPMD Execute")) {
        goto cleanup;
    }

    /* Await all device events */
    for (size_t d = 0; d < ndevs; d++) {
        if (spmd_await(ctx, events[d], "SPMD event"))
            goto cleanup;
        events[d] = NULL;
    }

    /* Allocate caller-visible output array: flat [ndevs * noutputs] tpu_buf_t* */
    tpu_buf_t** flat = (tpu_buf_t**)calloc(ndevs * noutputs, sizeof(tpu_buf_t*));
    if (!flat) goto oom;
    for (size_t d = 0; d < ndevs; d++) {
        for (size_t o = 0; o < noutputs; o++) {
            tpu_buf_t* b = (tpu_buf_t*)calloc(1, sizeof(tpu_buf_t));
            if (!b) goto oom;
            b->pjrt_buf        = pjrt_outs[d][o];
            b->ctx             = ctx;
            flat[d * noutputs + o] = b;
        }
    }
    *out_lists = flat;

    /* Free intermediate arrays */
    for (size_t d = 0; d < ndevs; d++) { free(inner[d]); free(pjrt_outs[d]); }
    free(inner); free(pjrt_outs); free(events);
    return 0;

oom:
    snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory");
cleanup:
    if (inner)     { for (size_t d = 0; d < ndevs; d++) free(inner[d]); free(inner); }
    if (pjrt_outs) { for (size_t d = 0; d < ndevs; d++) free(pjrt_outs[d]); free(pjrt_outs); }
    free(events);
    return -1;
}

/*
 * out_lists is the flat array [ndevs * noutputs] returned by tpu_run_spmd.
 * Access: out_lists[d * noutputs + o]
 */
void tpu_spmd_outputs_free(tpu_buf_t** out_lists,
                            size_t ndevs, size_t noutputs)
{
    if (!out_lists) return;
    for (size_t i = 0; i < ndevs * noutputs; i++)
        tpu_buf_free(out_lists[i]);
    free(out_lists);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* tpu_shard_f32                                                               */
/* ══════════════════════════════════════════════════════════════════════════ */

int tpu_shard_f32(tpu_ctx_t* ctx,
                  const float* data,
                  const int64_t* dims, size_t ndims,
                  size_t ndevs,
                  tpu_buf_t** shards)
{
    if (ndims == 0) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "shard: ndims must be >= 1");
        return -1;
    }
    if (dims[0] % (int64_t)ndevs != 0) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg),
                 "shard: dims[0]=%ld not divisible by ndevs=%zu", (long)dims[0], ndevs);
        return -1;
    }

    int64_t shard_rows = dims[0] / (int64_t)ndevs;

    /* Compute stride: product of dimensions excluding the outermost */
    size_t stride = 1;
    for (size_t i = 1; i < ndims; i++) stride *= (size_t)dims[i];

    /* Build shard dims: replace dims[0] with shard_rows */
    int64_t* shard_dims = (int64_t*)malloc(ndims * sizeof(int64_t));
    if (!shard_dims) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory");
        return -1;
    }
    shard_dims[0] = shard_rows;
    for (size_t i = 1; i < ndims; i++) shard_dims[i] = dims[i];

    for (size_t d = 0; d < ndevs; d++) {
        const float* shard_data = data + d * (size_t)shard_rows * stride;
        shards[d] = tpu_upload_f32(ctx, (int)d, shard_data, shard_dims, ndims);
        if (!shards[d]) {
            /* Free already-uploaded shards */
            for (size_t j = 0; j < d; j++) tpu_buf_free(shards[j]);
            free(shard_dims);
            return -1;
        }
    }
    free(shard_dims);
    return 0;
}
