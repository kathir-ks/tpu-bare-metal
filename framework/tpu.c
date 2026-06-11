/*
 * tpu.c — Core implementation of the TPU PJRT framework.
 *
 * Covers: init/destroy, device queries, upload, download, compile, run,
 * async run, device-to-device copy, topology string.
 */

#include "tpu.h"
#include "tpu_pjrt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <math.h>

/* ── Default libtpu.so search path ─────────────────────────────────────────── */

#define LIBTPU_DEFAULT_PATH \
    "/home/kathirks_gc/.local/lib/python3.10/site-packages/libtpu/libtpu.so"

/* ── Internal struct definitions ────────────────────────────────────────────── */

struct tpu_ctx_t {
    void*         dl_handle;
    void*         api_ptr;
    void**        fn;              /* fn = (void**)(api_ptr + 40) */
    PJRT_Client*  client;
    /* cached device lists — owned by PJRT, do not free */
    PJRT_Device** all_devices;
    size_t        num_all;
    PJRT_Device** addr_devices;
    size_t        num_addr;
    int           api_major;       /* plugin's PJRT C API version */
    int           api_minor;
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

/* ── Static init-time error buffer (for tpu_init failures) ─────────────────── */

static char g_init_errmsg[512];

/* ── Internal helpers ───────────────────────────────────────────────────────── */

/* Capture PJRT_Error into ctx->errmsg, destroy the error, return -1.
 * If err == NULL, returns 0 with no side effects. */
static int ctx_err(tpu_ctx_t* ctx, PJRT_Error* err, const char* where)
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

/* Await and destroy a PJRT_Event. Returns 0 on success, -1 on error. */
static int await_event(tpu_ctx_t* ctx, PJRT_Event* ev, const char* where)
{
    if (!ev) return 0;
    PJRT_Event_Await_Args aw;
    memset(&aw, 0, sizeof(aw));
    aw.struct_size = sizeof(aw);
    aw.event       = ev;
    int rc = ctx_err(ctx, PJRT_CALL(ctx->fn, FN_EVENT_AWAIT, PJRT_Event_Await_Args, &aw), where);
    PJRT_Event_Destroy_Args ed;
    memset(&ed, 0, sizeof(ed));
    ed.struct_size = sizeof(ed);
    ed.event       = ev;
    PJRT_CALLV(ctx->fn, FN_EVENT_DESTROY, PJRT_Event_Destroy_Args, &ed);
    return rc;
}

/* Read a file into a malloc'd buffer. *out_size is set.
 * Returns NULL and prints an error on failure. */
static char* read_file(const char* path, size_t* out_size)
{
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

/* ── tpu_dtype helpers ──────────────────────────────────────────────────────── */

size_t tpu_dtype_size(tpu_dtype_t dtype)
{
    switch (dtype) {
    case TPU_DTYPE_PRED: return 1;
    case TPU_DTYPE_S8: case TPU_DTYPE_U8: return 1;
    case TPU_DTYPE_S16: case TPU_DTYPE_U16: case TPU_DTYPE_F16: case TPU_DTYPE_BF16: return 2;
    case TPU_DTYPE_S32: case TPU_DTYPE_U32: case TPU_DTYPE_F32: return 4;
    case TPU_DTYPE_S64: case TPU_DTYPE_U64: case TPU_DTYPE_F64: case TPU_DTYPE_C64: return 8;
    case TPU_DTYPE_C128: return 16;
    default: return 0;
    }
}

const char* tpu_dtype_name(tpu_dtype_t dtype)
{
    switch (dtype) {
    case TPU_DTYPE_PRED:  return "pred";
    case TPU_DTYPE_S8:    return "s8";
    case TPU_DTYPE_S16:   return "s16";
    case TPU_DTYPE_S32:   return "s32";
    case TPU_DTYPE_S64:   return "s64";
    case TPU_DTYPE_U8:    return "u8";
    case TPU_DTYPE_U16:   return "u16";
    case TPU_DTYPE_U32:   return "u32";
    case TPU_DTYPE_U64:   return "u64";
    case TPU_DTYPE_F16:   return "f16";
    case TPU_DTYPE_F32:   return "f32";
    case TPU_DTYPE_F64:   return "f64";
    case TPU_DTYPE_BF16:  return "bf16";
    case TPU_DTYPE_C64:   return "c64";
    case TPU_DTYPE_C128:  return "c128";
    default:              return "?";
    }
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Init / teardown                                                             */
/* ══════════════════════════════════════════════════════════════════════════ */

tpu_ctx_t* tpu_init(const char* lib_path)
{
    /* Any PJRT plugin works (libtpu.so, xla_cuda_plugin.so, CPU plugin, …):
     * explicit path → PJRT_PLUGIN_PATH → LIBTPU_PATH (legacy) → default. */
    if (!lib_path) lib_path = getenv("PJRT_PLUGIN_PATH");
    if (!lib_path) lib_path = getenv("LIBTPU_PATH");
    if (!lib_path) lib_path = LIBTPU_DEFAULT_PATH;

    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        snprintf(g_init_errmsg, sizeof(g_init_errmsg), "dlopen(%s): %s", lib_path, dlerror());
        return NULL;
    }

    typedef void* (*GetPjrtApiFn)(void);
    GetPjrtApiFn get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) {
        snprintf(g_init_errmsg, sizeof(g_init_errmsg), "GetPjrtApi not found in %s", lib_path);
        dlclose(handle);
        return NULL;
    }

    void* api_ptr = get_api();
    void** fn     = (void**)((char*)api_ptr + 40);

    /* Header: struct_size(8) ext(8) version{ss(8) ext(8) major(4) minor(4)}.
     * The function table is append-only across API versions, so existing
     * indices stay valid — but the plugin's table must at least cover the
     * highest index we call. */
    size_t api_struct_size = *(size_t*)api_ptr;
    int api_major = *(int*)((char*)api_ptr + 32);
    int api_minor = *(int*)((char*)api_ptr + 36);
    size_t need = 40 + 8 * (FN_BUF_TOHOST + 1);
    if (api_struct_size < need) {
        snprintf(g_init_errmsg, sizeof(g_init_errmsg),
                 "%s: PJRT API v%d.%d table too small (%zu bytes, need %zu) — plugin too old",
                 lib_path, api_major, api_minor, api_struct_size, need);
        dlclose(handle);
        return NULL;
    }

    /* Plugin_Initialize */
    PJRT_Plugin_Initialize_Args pi;
    memset(&pi, 0, sizeof(pi));
    pi.struct_size = sizeof(pi);
    PJRT_Error* err = PJRT_CALL(fn, FN_PLUGIN_INITIALIZE, PJRT_Plugin_Initialize_Args, &pi);
    if (err) {
        /* Extract message without a ctx */
        PJRT_Error_Message_Args ma;
        memset(&ma, 0, sizeof(ma));
        ma.struct_size = sizeof(ma);
        ma.error       = err;
        PJRT_CALLV(fn, FN_ERROR_MESSAGE, PJRT_Error_Message_Args, &ma);
        snprintf(g_init_errmsg, sizeof(g_init_errmsg),
                 "[PluginInit] %.*s", (int)ma.message_size, ma.message);
        PJRT_Error_Destroy_Args da;
        memset(&da, 0, sizeof(da));
        da.struct_size = sizeof(da);
        da.error       = err;
        PJRT_CALLV(fn, FN_ERROR_DESTROY, PJRT_Error_Destroy_Args, &da);
        dlclose(handle);
        return NULL;
    }

    /* Client_Create */
    PJRT_Client_Create_Args cc;
    memset(&cc, 0, sizeof(cc));
    cc.struct_size = sizeof(cc);
    err = PJRT_CALL(fn, FN_CLIENT_CREATE, PJRT_Client_Create_Args, &cc);
    if (err) {
        PJRT_Error_Message_Args ma;
        memset(&ma, 0, sizeof(ma));
        ma.struct_size = sizeof(ma);
        ma.error       = err;
        PJRT_CALLV(fn, FN_ERROR_MESSAGE, PJRT_Error_Message_Args, &ma);
        snprintf(g_init_errmsg, sizeof(g_init_errmsg),
                 "[ClientCreate] %.*s", (int)ma.message_size, ma.message);
        PJRT_Error_Destroy_Args da;
        memset(&da, 0, sizeof(da));
        da.struct_size = sizeof(da);
        da.error       = err;
        PJRT_CALLV(fn, FN_ERROR_DESTROY, PJRT_Error_Destroy_Args, &da);
        dlclose(handle);
        return NULL;
    }
    PJRT_Client* client = cc.client;

    /* Cache device lists */
    PJRT_Client_Devices_Args dv;
    memset(&dv, 0, sizeof(dv));
    dv.struct_size = sizeof(dv);
    dv.client      = client;
    PJRT_CALL(fn, FN_CLIENT_DEVICES, PJRT_Client_Devices_Args, &dv);

    PJRT_Client_AddressableDevices_Args adv;
    memset(&adv, 0, sizeof(adv));
    adv.struct_size = sizeof(adv);
    adv.client      = client;
    PJRT_CALL(fn, FN_CLIENT_ADDRDEVICES, PJRT_Client_AddressableDevices_Args, &adv);

    tpu_ctx_t* ctx = (tpu_ctx_t*)calloc(1, sizeof(tpu_ctx_t));
    if (!ctx) {
        snprintf(g_init_errmsg, sizeof(g_init_errmsg), "out of memory");
        dlclose(handle);
        return NULL;
    }
    ctx->dl_handle   = handle;
    ctx->api_ptr     = api_ptr;
    ctx->fn          = fn;
    ctx->client      = client;
    ctx->all_devices  = (PJRT_Device**)dv.devices;
    ctx->num_all      = dv.num_devices;
    ctx->addr_devices = (PJRT_Device**)adv.addressable_devices;
    ctx->num_addr     = adv.num_addressable_devices;
    ctx->api_major    = api_major;
    ctx->api_minor    = api_minor;
    return ctx;
}

void tpu_api_version(tpu_ctx_t* ctx, int* major, int* minor)
{
    if (major) *major = ctx ? ctx->api_major : 0;
    if (minor) *minor = ctx ? ctx->api_minor : 0;
}

void tpu_destroy(tpu_ctx_t* ctx)
{
    if (!ctx) return;
    PJRT_Client_Destroy_Args cd;
    memset(&cd, 0, sizeof(cd));
    cd.struct_size = sizeof(cd);
    cd.client      = ctx->client;
    PJRT_CALLV(ctx->fn, FN_CLIENT_DESTROY, PJRT_Client_Destroy_Args, &cd);
    dlclose(ctx->dl_handle);
    free(ctx);
}

const char* tpu_strerror(tpu_ctx_t* ctx)
{
    if (!ctx) return g_init_errmsg[0] ? g_init_errmsg : NULL;
    return ctx->errmsg[0] ? ctx->errmsg : NULL;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Device queries                                                              */
/* ══════════════════════════════════════════════════════════════════════════ */

int tpu_num_devices(tpu_ctx_t* ctx)             { return (int)ctx->num_all; }
int tpu_num_addressable_devices(tpu_ctx_t* ctx) { return (int)ctx->num_addr; }

int tpu_device_info(tpu_ctx_t* ctx, int dev_idx, tpu_device_info_t* out)
{
    if (dev_idx < 0 || (size_t)dev_idx >= ctx->num_addr) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg),
                 "device index %d out of range [0,%zu)", dev_idx, ctx->num_addr);
        return -1;
    }
    PJRT_Device* dev = ctx->addr_devices[dev_idx];
    memset(out, 0, sizeof(*out));
    out->bytes_limit = -1;

    /* Get description handle */
    PJRT_Device_GetDescription_Args gd;
    memset(&gd, 0, sizeof(gd));
    gd.struct_size = sizeof(gd);
    gd.device      = dev;
    if (ctx_err(ctx, PJRT_CALL(ctx->fn, FN_DEV_GETDESC, PJRT_Device_GetDescription_Args, &gd), "GetDesc"))
        return -1;
    PJRT_DeviceDescription* dd = gd.device_description;

    /* ID */
    PJRT_DeviceDescription_Id_Args id_a;
    memset(&id_a, 0, sizeof(id_a));
    id_a.struct_size         = sizeof(id_a);
    id_a.device_description  = dd;
    PJRT_CALL(ctx->fn, FN_DEVDESC_ID, PJRT_DeviceDescription_Id_Args, &id_a);
    out->id = (int)id_a.id;

    /* Kind */
    PJRT_DeviceDescription_Kind_Args kd;
    memset(&kd, 0, sizeof(kd));
    kd.struct_size        = sizeof(kd);
    kd.device_description = dd;
    PJRT_CALL(ctx->fn, FN_DEVDESC_KIND, PJRT_DeviceDescription_Kind_Args, &kd);
    snprintf(out->kind, sizeof(out->kind), "%.*s", (int)kd.device_kind_size, kd.device_kind);

    /* ToString */
    PJRT_DeviceDescription_ToString_Args ts;
    memset(&ts, 0, sizeof(ts));
    ts.struct_size        = sizeof(ts);
    ts.device_description = dd;
    PJRT_CALL(ctx->fn, FN_DEVDESC_TOSTRING, PJRT_DeviceDescription_ToString_Args, &ts);
    snprintf(out->description, sizeof(out->description), "%.*s",
             (int)ts.to_string_size, ts.to_string);

    /* IsAddressable */
    PJRT_Device_IsAddressable_Args ia;
    memset(&ia, 0, sizeof(ia));
    ia.struct_size = sizeof(ia);
    ia.device      = dev;
    PJRT_CALL(ctx->fn, FN_DEV_ISADDR, PJRT_Device_IsAddressable_Args, &ia);
    out->is_addressable = ia.is_addressable;

    /* LocalHardwareId */
    PJRT_Device_LocalHardwareId_Args hw;
    memset(&hw, 0, sizeof(hw));
    hw.struct_size = sizeof(hw);
    hw.device      = dev;
    PJRT_CALL(ctx->fn, FN_DEV_LOCALHWID, PJRT_Device_LocalHardwareId_Args, &hw);
    out->local_hw_id = hw.local_hardware_id;

    /* MemoryStats (optional — ignore errors) */
    PJRT_Device_MemoryStats_Args ms;
    memset(&ms, 0, sizeof(ms));
    ms.struct_size = sizeof(ms);
    ms.device      = dev;
    PJRT_Error* merr = PJRT_CALL(ctx->fn, FN_DEV_MEMSTATS, PJRT_Device_MemoryStats_Args, &ms);
    if (!merr) {
        out->bytes_in_use      = ms.bytes_in_use;
        out->bytes_limit_valid = ms.bytes_limit_is_set;
        if (ms.bytes_limit_is_set) out->bytes_limit = ms.bytes_limit;
    } else {
        /* Not all TPU versions support MemoryStats; silently ignore */
        PJRT_Error_Destroy_Args da;
        memset(&da, 0, sizeof(da));
        da.struct_size = sizeof(da);
        da.error       = merr;
        PJRT_CALLV(ctx->fn, FN_ERROR_DESTROY, PJRT_Error_Destroy_Args, &da);
    }
    return 0;
}

char* tpu_topology_string(tpu_ctx_t* ctx)
{
    /* Get platform name and version from the client directly */
    PJRT_Client_PlatformName_Args pn;
    memset(&pn, 0, sizeof(pn));
    pn.struct_size = sizeof(pn);
    pn.client      = ctx->client;
    if (ctx_err(ctx, PJRT_CALL(ctx->fn, FN_CLIENT_PLATFORMNAME,
                               PJRT_Client_PlatformName_Args, &pn), "PlatformName"))
        return NULL;

    PJRT_Client_PlatformVersion_Args pv;
    memset(&pv, 0, sizeof(pv));
    pv.struct_size = sizeof(pv);
    pv.client      = ctx->client;
    PJRT_CALL(ctx->fn, FN_CLIENT_PLATFORMVERSION, PJRT_Client_PlatformVersion_Args, &pv);

    PJRT_Client_ProcessIndex_Args pi;
    memset(&pi, 0, sizeof(pi));
    pi.struct_size = sizeof(pi);
    pi.client      = ctx->client;
    PJRT_CALL(ctx->fn, FN_CLIENT_PROCESSINDEX, PJRT_Client_ProcessIndex_Args, &pi);

    size_t sz = 512 + pn.platform_name_size + pv.platform_version_size;
    char* out = (char*)malloc(sz);
    if (!out) return NULL;
    snprintf(out, sz,
             "Platform:      %.*s\n"
             "Version:       %.*s\n"
             "Process index: %d\n"
             "All devices:   %zu\n"
             "Addressable:   %zu",
             (int)pn.platform_name_size,    pn.platform_name,
             (int)pv.platform_version_size, pv.platform_version,
             pi.process_index,
             ctx->num_all,
             ctx->num_addr);
    return out;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Compile                                                                     */
/* ══════════════════════════════════════════════════════════════════════════ */

tpu_exec_t* tpu_compile_file(tpu_ctx_t* ctx,
                              const char* hlo_pb_path,
                              const char* opts_pb_path)
{
    size_t code_sz = 0;
    char*  code    = read_file(hlo_pb_path, &code_sz);
    if (!code) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "cannot read %s", hlo_pb_path);
        return NULL;
    }

    size_t opts_sz = 0;
    char*  opts    = NULL;
    if (opts_pb_path) {
        opts = read_file(opts_pb_path, &opts_sz);
        if (!opts) {
            snprintf(ctx->errmsg, sizeof(ctx->errmsg), "cannot read %s", opts_pb_path);
            free(code);
            return NULL;
        }
    }

    tpu_exec_t* exec = tpu_compile_buf(ctx, code, code_sz, "hlo", opts, opts_sz);
    free(code);
    free(opts);  /* safe: free(NULL) is a no-op */
    return exec;
}

tpu_exec_t* tpu_compile_buf(tpu_ctx_t* ctx,
                             const void* code, size_t code_sz,
                             const char* format,
                             const void* opts, size_t opts_sz)
{
    PJRT_Program prog;
    memset(&prog, 0, sizeof(prog));
    prog.struct_size = sizeof(prog);
    prog.code        = (char*)(uintptr_t)code;
    prog.code_size   = code_sz;
    prog.format      = format;
    prog.format_size = strlen(format);

    PJRT_Client_Compile_Args cc;
    memset(&cc, 0, sizeof(cc));
    cc.struct_size          = sizeof(cc);
    cc.client               = ctx->client;
    cc.program              = &prog;
    cc.compile_options      = (const char*)opts;
    cc.compile_options_size = opts_sz;

    if (ctx_err(ctx, PJRT_CALL(ctx->fn, FN_CLIENT_COMPILE,
                               PJRT_Client_Compile_Args, &cc), "Compile"))
        return NULL;

    PJRT_LoadedExecutable* lexec = cc.executable;

    /* Get PJRT_Executable to query num_outputs */
    PJRT_LoadedExecutable_GetExecutable_Args ge;
    memset(&ge, 0, sizeof(ge));
    ge.struct_size        = sizeof(ge);
    ge.loaded_executable  = lexec;
    if (ctx_err(ctx, PJRT_CALL(ctx->fn, FN_LEXEC_GETEXEC,
                               PJRT_LoadedExecutable_GetExecutable_Args, &ge), "GetExec")) {
        PJRT_LoadedExecutable_Destroy_Args ld;
        memset(&ld, 0, sizeof(ld));
        ld.struct_size = sizeof(ld);
        ld.executable  = lexec;
        PJRT_CALLV(ctx->fn, FN_LEXEC_DESTROY, PJRT_LoadedExecutable_Destroy_Args, &ld);
        return NULL;
    }

    PJRT_Executable_NumOutputs_Args no;
    memset(&no, 0, sizeof(no));
    no.struct_size = sizeof(no);
    no.executable  = ge.executable;
    ctx_err(ctx, PJRT_CALL(ctx->fn, FN_EXEC_NUMOUTPUTS,
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

int tpu_exec_num_outputs(tpu_exec_t* exec)
{
    return exec ? (int)exec->num_outputs : -1;
}

int tpu_exec_device_order(tpu_exec_t* exec, int* out_dev_indices, size_t n)
{
    tpu_ctx_t* ctx = exec->ctx;
    PJRT_LoadedExecutable_AddressableDevices_Args ad;
    memset(&ad, 0, sizeof(ad));
    ad.struct_size = sizeof(ad);
    ad.executable  = exec->lexec;
    if (ctx_err(ctx, PJRT_CALL(ctx->fn, FN_LEXEC_ADDRDEVICES,
                               PJRT_LoadedExecutable_AddressableDevices_Args, &ad),
                "LoadedExecutable_AddressableDevices"))
        return -1;
    for (size_t r = 0; r < n && r < ad.num_addressable_devices; r++) {
        PJRT_Device* dev = (PJRT_Device*)ad.addressable_devices[r];
        out_dev_indices[r] = -1;
        for (size_t i = 0; i < ctx->num_addr; i++) {
            if (ctx->addr_devices[i] == dev) { out_dev_indices[r] = (int)i; break; }
        }
    }
    return 0;
}

void tpu_exec_free(tpu_exec_t* exec)
{
    if (!exec) return;
    PJRT_LoadedExecutable_Destroy_Args ld;
    memset(&ld, 0, sizeof(ld));
    ld.struct_size = sizeof(ld);
    ld.executable  = exec->lexec;
    PJRT_CALLV(exec->ctx->fn, FN_LEXEC_DESTROY, PJRT_LoadedExecutable_Destroy_Args, &ld);
    free(exec);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Buffer upload  (host → TPU HBM)                                            */
/* ══════════════════════════════════════════════════════════════════════════ */

tpu_buf_t* tpu_upload(tpu_ctx_t* ctx, int dev_idx,
                      const void* data, tpu_dtype_t dtype,
                      const int64_t* dims, size_t ndims)
{
    if (dev_idx < 0 || (size_t)dev_idx >= ctx->num_addr) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg),
                 "device index %d out of range", dev_idx);
        return NULL;
    }

    PJRT_Client_BufferFromHostBuffer_Args bh;
    memset(&bh, 0, sizeof(bh));
    bh.struct_size            = sizeof(bh);
    bh.client                 = ctx->client;
    bh.data                   = data;
    bh.type                   = (int)dtype;
    bh.dims                   = dims;
    bh.num_dims               = ndims;
    bh.host_buffer_semantics  = PJRT_HostBuf_ImmutableOnlyDuringCall;
    bh.device                 = ctx->addr_devices[dev_idx];

    if (ctx_err(ctx, PJRT_CALL(ctx->fn, FN_CLIENT_BUFFROMHOST,
                               PJRT_Client_BufferFromHostBuffer_Args, &bh), "BufFromHost"))
        return NULL;

    if (await_event(ctx, bh.done_with_host_buffer, "BufFromHost-event"))
        return NULL;

    tpu_buf_t* buf = (tpu_buf_t*)calloc(1, sizeof(tpu_buf_t));
    if (!buf) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory");
        return NULL;
    }
    buf->pjrt_buf = bh.buffer;
    buf->ctx      = ctx;
    return buf;
}

tpu_buf_t* tpu_upload_f32(tpu_ctx_t* ctx, int dev_idx,
                           const float* data,
                           const int64_t* dims, size_t ndims)
{
    return tpu_upload(ctx, dev_idx, data, TPU_DTYPE_F32, dims, ndims);
}

tpu_buf_t* tpu_upload_bf16(tpu_ctx_t* ctx, int dev_idx,
                            const void* data,
                            const int64_t* dims, size_t ndims)
{
    return tpu_upload(ctx, dev_idx, data, TPU_DTYPE_BF16, dims, ndims);
}

tpu_buf_t* tpu_upload_s32(tpu_ctx_t* ctx, int dev_idx,
                           const int32_t* data,
                           const int64_t* dims, size_t ndims)
{
    return tpu_upload(ctx, dev_idx, (const void*)data, TPU_DTYPE_S32, dims, ndims);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Buffer download  (TPU HBM → host)                                          */
/* ══════════════════════════════════════════════════════════════════════════ */

int tpu_download(tpu_buf_t* buf, void* dst, size_t dst_size)
{
    /* Query rank so we can request a dense row-major host layout. Without an
     * explicit host_layout, libtpu returns the on-device (tiled/padded) layout
     * for shapes whose minor dim isn't 128-aligned, scrambling the result. */
    int64_t dims[8];
    size_t  ndims = 0;
    if (tpu_buf_dims(buf, dims, &ndims) != 0) return -1;

    int64_t minor_to_major[8];
    for (size_t i = 0; i < ndims; i++)
        minor_to_major[i] = (int64_t)(ndims - 1 - i);  /* row-major: last dim minor */

    PJRT_Buffer_MemoryLayout layout;
    memset(&layout, 0, sizeof(layout));
    layout.struct_size              = sizeof(layout);
    layout.type                     = PJRT_Buffer_MemoryLayout_Type_Tiled;
    layout.tiled.struct_size        = sizeof(layout.tiled);
    layout.tiled.minor_to_major     = minor_to_major;
    layout.tiled.minor_to_major_size = ndims;
    layout.tiled.num_tiles          = 0;

    PJRT_Buffer_ToHostBuffer_Args th;
    memset(&th, 0, sizeof(th));
    th.struct_size = sizeof(th);
    th.src         = buf->pjrt_buf;
    th.host_layout = (ndims > 0) ? &layout : NULL;
    th.dst         = dst;
    th.dst_size    = dst_size;

    if (ctx_err(buf->ctx,
                PJRT_CALL(buf->ctx->fn, FN_BUF_TOHOST,
                          PJRT_Buffer_ToHostBuffer_Args, &th), "BufToHost"))
        return -1;

    return await_event(buf->ctx, th.event, "BufToHost-event");
}

size_t tpu_buf_size(tpu_buf_t* buf)
{
    PJRT_Buffer_OnDeviceSizeInBytes_Args sz;
    memset(&sz, 0, sizeof(sz));
    sz.struct_size = sizeof(sz);
    sz.buffer      = buf->pjrt_buf;
    if (ctx_err(buf->ctx, PJRT_CALL(buf->ctx->fn, FN_BUF_ONDEVSZ,
                                    PJRT_Buffer_OnDeviceSizeInBytes_Args, &sz), "BufSize"))
        return 0;
    return sz.on_device_size_in_bytes;
}

int tpu_buf_dims(tpu_buf_t* buf, int64_t* dims, size_t* ndims)
{
    PJRT_Buffer_Dimensions_Args bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.buffer      = buf->pjrt_buf;
    if (ctx_err(buf->ctx, PJRT_CALL(buf->ctx->fn, FN_BUF_DIMS,
                                    PJRT_Buffer_Dimensions_Args, &bd), "BufDims"))
        return -1;
    if (dims && ndims) {
        size_t n = bd.num_dims < 8 ? bd.num_dims : 8;
        for (size_t i = 0; i < n; i++) dims[i] = bd.dims[i];
    }
    if (ndims) *ndims = bd.num_dims;
    return 0;
}

tpu_dtype_t tpu_buf_dtype(tpu_buf_t* buf)
{
    PJRT_Buffer_ElementType_Args et;
    memset(&et, 0, sizeof(et));
    et.struct_size = sizeof(et);
    et.buffer      = buf->pjrt_buf;
    ctx_err(buf->ctx, PJRT_CALL(buf->ctx->fn, FN_BUF_ELEMTYPE,
                                PJRT_Buffer_ElementType_Args, &et), "BufElemType");
    return (tpu_dtype_t)et.element_type;
}

void tpu_buf_free(tpu_buf_t* buf)
{
    if (!buf) return;
    PJRT_Buffer_Destroy_Args bd;
    memset(&bd, 0, sizeof(bd));
    bd.struct_size = sizeof(bd);
    bd.buffer      = buf->pjrt_buf;
    PJRT_CALLV(buf->ctx->fn, FN_BUF_DESTROY, PJRT_Buffer_Destroy_Args, &bd);
    free(buf);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Single-device execution (internal helper)                                  */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Build PJRT_Buffer* arrays and call Execute. Callers handle await/async. */
static int run_execute(tpu_exec_t* exec, int dev_idx,
                       tpu_buf_t* const* args, size_t nargs,
                       tpu_buf_t** outputs, size_t noutputs,
                       PJRT_Event** out_event)
{
    tpu_ctx_t* ctx = exec->ctx;
    if (dev_idx < 0 || (size_t)dev_idx >= ctx->num_addr) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg),
                 "device index %d out of range", dev_idx);
        return -1;
    }

    /* Build inner arg array (PJRT_Buffer* per arg) */
    PJRT_Buffer** inner = (PJRT_Buffer**)alloca(nargs * sizeof(PJRT_Buffer*));
    for (size_t i = 0; i < nargs; i++) inner[i] = args[i]->pjrt_buf;

    PJRT_Buffer* const* arg_list_ptrs[] = { (PJRT_Buffer* const*)inner };

    /* Build output array (PJRT_Buffer* per output, filled by PJRT) */
    PJRT_Buffer** out_ptrs = (PJRT_Buffer**)calloc(noutputs, sizeof(PJRT_Buffer*));
    if (!out_ptrs) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory");
        return -1;
    }
    PJRT_Buffer** out_list_ptrs[] = { out_ptrs };

    PJRT_ExecuteOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = sizeof(opts);

    PJRT_Event* completion = NULL;

    PJRT_LoadedExecutable_Execute_Args ex;
    memset(&ex, 0, sizeof(ex));
    ex.struct_size           = sizeof(ex);
    ex.executable            = exec->lexec;
    ex.options               = &opts;
    ex.argument_lists        = (PJRT_Buffer* const* const*)arg_list_ptrs;
    ex.num_devices           = 1;
    ex.num_args              = nargs;
    ex.output_lists          = (PJRT_Buffer** const*)out_list_ptrs;
    ex.device_complete_events = &completion;
    ex.execute_device        = ctx->addr_devices[dev_idx];

    int rc = ctx_err(ctx, PJRT_CALL(ctx->fn, FN_LEXEC_EXECUTE,
                                    PJRT_LoadedExecutable_Execute_Args, &ex), "Execute");
    if (rc) {
        free(out_ptrs);
        return -1;
    }

    /* Wrap output buffers into tpu_buf_t */
    for (size_t i = 0; i < noutputs; i++) {
        tpu_buf_t* b = (tpu_buf_t*)calloc(1, sizeof(tpu_buf_t));
        if (!b) { snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory"); free(out_ptrs); return -1; }
        b->pjrt_buf = out_ptrs[i];
        b->ctx      = ctx;
        outputs[i]  = b;
    }
    free(out_ptrs);

    *out_event = completion;
    return 0;
}

/* ── tpu_run (synchronous) ───────────────────────────────────────────────── */

int tpu_run(tpu_exec_t* exec, int dev_idx,
            tpu_buf_t* const* args, size_t nargs,
            tpu_buf_t** outputs, size_t noutputs)
{
    PJRT_Event* ev = NULL;
    if (run_execute(exec, dev_idx, args, nargs, outputs, noutputs, &ev))
        return -1;
    return await_event(exec->ctx, ev, "Execute-event");
}

tpu_buf_t* tpu_run1(tpu_exec_t* exec, tpu_buf_t* a)
{
    tpu_buf_t* out = NULL;
    tpu_buf_t* const args[] = { a };
    if (tpu_run(exec, 0, args, 1, &out, 1)) return NULL;
    return out;
}

tpu_buf_t* tpu_run2(tpu_exec_t* exec, tpu_buf_t* a, tpu_buf_t* b)
{
    tpu_buf_t* out = NULL;
    tpu_buf_t* const args[] = { a, b };
    if (tpu_run(exec, 0, args, 2, &out, 1)) return NULL;
    return out;
}

/* ── tpu_run_async ───────────────────────────────────────────────────────── */

typedef struct {
    tpu_on_ready_fn  cb;
    void*            user_arg;
    tpu_ctx_t*       ctx;
} async_bridge_t;

static void pjrt_async_callback(PJRT_Error* error, void* user_arg)
{
    async_bridge_t* bridge = (async_bridge_t*)user_arg;
    if (error) {
        PJRT_Error_Message_Args ma;
        memset(&ma, 0, sizeof(ma));
        ma.struct_size = sizeof(ma);
        ma.error       = error;
        PJRT_CALLV(bridge->ctx->fn, FN_ERROR_MESSAGE, PJRT_Error_Message_Args, &ma);
        /* Copy message to static buffer — the error is about to be destroyed */
        static char async_errmsg[512];
        snprintf(async_errmsg, sizeof(async_errmsg),
                 "%.*s", (int)ma.message_size, ma.message);
        PJRT_Error_Destroy_Args da;
        memset(&da, 0, sizeof(da));
        da.struct_size = sizeof(da);
        da.error       = error;
        PJRT_CALLV(bridge->ctx->fn, FN_ERROR_DESTROY, PJRT_Error_Destroy_Args, &da);
        bridge->cb(async_errmsg, bridge->user_arg);
    } else {
        bridge->cb(NULL, bridge->user_arg);
    }
    free(bridge);
}

int tpu_run_async(tpu_exec_t* exec, int dev_idx,
                  tpu_buf_t* const* args, size_t nargs,
                  tpu_buf_t** outputs, size_t noutputs,
                  tpu_on_ready_fn cb, void* user_arg)
{
    PJRT_Event* ev = NULL;
    if (run_execute(exec, dev_idx, args, nargs, outputs, noutputs, &ev))
        return -1;

    if (!ev) {
        /* Completed synchronously during Execute */
        cb(NULL, user_arg);
        return 0;
    }

    async_bridge_t* bridge = (async_bridge_t*)malloc(sizeof(async_bridge_t));
    if (!bridge) {
        await_event(exec->ctx, ev, "async-fallback");
        cb(NULL, user_arg);
        return 0;
    }
    bridge->cb       = cb;
    bridge->user_arg = user_arg;
    bridge->ctx      = exec->ctx;

    PJRT_Event_OnReady_Args or_a;
    memset(&or_a, 0, sizeof(or_a));
    or_a.struct_size = sizeof(or_a);
    or_a.event       = ev;
    or_a.callback    = pjrt_async_callback;
    or_a.user_arg    = bridge;

    int rc = ctx_err(exec->ctx,
                     PJRT_CALL(exec->ctx->fn, FN_EVENT_ONREADY,
                               PJRT_Event_OnReady_Args, &or_a), "EventOnReady");
    if (rc) {
        free(bridge);
        return -1;
    }
    /* bridge and event are now owned by the PJRT callback machinery */
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════ */
/* Device-to-device copy                                                       */
/* ══════════════════════════════════════════════════════════════════════════ */

tpu_buf_t* tpu_buf_copy_to_device(tpu_buf_t* src, int dst_dev_idx)
{
    tpu_ctx_t* ctx = src->ctx;
    if (dst_dev_idx < 0 || (size_t)dst_dev_idx >= ctx->num_addr) {
        snprintf(ctx->errmsg, sizeof(ctx->errmsg),
                 "dst device index %d out of range", dst_dev_idx);
        return NULL;
    }

    PJRT_Buffer_CopyToDevice_Args cd;
    memset(&cd, 0, sizeof(cd));
    cd.struct_size = sizeof(cd);
    cd.buffer      = src->pjrt_buf;
    cd.dst_device  = ctx->addr_devices[dst_dev_idx];

    if (ctx_err(ctx, PJRT_CALL(ctx->fn, FN_BUF_COPYTODEV,
                               PJRT_Buffer_CopyToDevice_Args, &cd), "CopyToDevice"))
        return NULL;

    tpu_buf_t* dst = (tpu_buf_t*)calloc(1, sizeof(tpu_buf_t));
    if (!dst) { snprintf(ctx->errmsg, sizeof(ctx->errmsg), "out of memory"); return NULL; }
    dst->pjrt_buf = cd.dst_buffer;
    dst->ctx      = ctx;
    return dst;
}
