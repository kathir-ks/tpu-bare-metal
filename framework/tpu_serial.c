/*
 * tpu_serial.c — Executable serialization: save to disk and load back.
 *
 * The serialized format is TPU-runtime-specific and not portable across
 * different libtpu.so versions or TPU hardware generations.
 */

#include "tpu.h"
#include "tpu_pjrt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Access ctx/exec internals (same layout as tpu.c) */
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

static int serial_err(tpu_ctx_t* ctx, PJRT_Error* err, const char* where)
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

/* ══════════════════════════════════════════════════════════════════════════ */
/* tpu_exec_save                                                               */
/* ══════════════════════════════════════════════════════════════════════════ */

int tpu_exec_save(tpu_exec_t* exec, const char* path)
{
    tpu_ctx_t* ctx = exec->ctx;

    /* Get PJRT_Executable* from LoadedExecutable */
    PJRT_LoadedExecutable_GetExecutable_Args ge;
    memset(&ge, 0, sizeof(ge));
    ge.struct_size       = sizeof(ge);
    ge.loaded_executable = exec->lexec;
    if (serial_err(ctx, PJRT_CALL(ctx->fn, FN_LEXEC_GETEXEC,
                                  PJRT_LoadedExecutable_GetExecutable_Args, &ge), "GetExec"))
        return -1;

    /* Serialize */
    PJRT_Executable_Serialize_Args sa;
    memset(&sa, 0, sizeof(sa));
    sa.struct_size = sizeof(sa);
    sa.executable  = ge.executable;
    if (serial_err(ctx, PJRT_CALL(ctx->fn, FN_EXEC_SERIALIZE,
                                  PJRT_Executable_Serialize_Args, &sa), "Serialize"))
        return -1;

    /* Write to file */
    FILE* f = fopen(path, "wb");
    if (!f) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "cannot open %s for writing", path);
        sa.serialized_executable_deleter(sa.serialized_executable);
        return -1;
    }
    size_t written = fwrite(sa.serialized_bytes, 1, sa.serialized_bytes_size, f);
    int ferr = fclose(f);
    sa.serialized_executable_deleter(sa.serialized_executable);

    if (written != sa.serialized_bytes_size || ferr) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg),
                 "write error for %s (wrote %zu of %zu bytes)",
                 path, written, sa.serialized_bytes_size);
        return -1;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* tpu_exec_load                                                               */
/* ══════════════════════════════════════════════════════════════════════════ */

tpu_exec_t* tpu_exec_load(tpu_ctx_t* ctx, const char* path)
{
    /* Read serialized bytes */
    FILE* f = fopen(path, "rb");
    if (!f) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "cannot open %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* bytes = (char*)malloc((size_t)fsz);
    if (!bytes) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory");
        fclose(f);
        return NULL;
    }
    if (fread(bytes, 1, (size_t)fsz, f) != (size_t)fsz) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "read error for %s", path);
        free(bytes); fclose(f);
        return NULL;
    }
    fclose(f);

    /* Deserialize and load */
    PJRT_Executable_DeserializeAndLoad_Args dl;
    memset(&dl, 0, sizeof(dl));
    dl.struct_size                = sizeof(dl);
    dl.client                     = ctx->client;
    dl.serialized_executable      = bytes;
    dl.serialized_executable_size = (size_t)fsz;
    /* overridden_serialized_compile_options = NULL: use embedded options */

    if (serial_err(ctx, PJRT_CALL(ctx->fn, FN_EXEC_DESERIALIZE_LOAD,
                                  PJRT_Executable_DeserializeAndLoad_Args, &dl), "Deserialize")) {
        free(bytes);
        return NULL;
    }
    free(bytes);

    PJRT_LoadedExecutable* lexec = dl.loaded_executable;

    /* Cache num_outputs */
    PJRT_LoadedExecutable_GetExecutable_Args ge;
    memset(&ge, 0, sizeof(ge));
    ge.struct_size       = sizeof(ge);
    ge.loaded_executable = lexec;
    serial_err(ctx, PJRT_CALL(ctx->fn, FN_LEXEC_GETEXEC,
                              PJRT_LoadedExecutable_GetExecutable_Args, &ge), "GetExec");

    PJRT_Executable_NumOutputs_Args no;
    memset(&no, 0, sizeof(no));
    no.struct_size = sizeof(no);
    no.executable  = ge.executable;
    serial_err(ctx, PJRT_CALL(ctx->fn, FN_EXEC_NUMOUTPUTS,
                              PJRT_Executable_NumOutputs_Args, &no), "NumOutputs");

    tpu_exec_t* exec = (tpu_exec_t*)calloc(1, sizeof(tpu_exec_t));
    if (!exec) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory");
        return NULL;
    }
    exec->lexec       = lexec;
    exec->ctx         = ctx;
    exec->num_outputs = no.num_outputs;
    return exec;
}
