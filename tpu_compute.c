/*
 * TPU Computation via libtpu.so PJRT C API
 * =========================================
 *
 * Compiles an HLO program and executes matrix multiply on TPU.
 * Also demonstrates element-wise add.
 *
 * Compile: gcc -o tpu_compute tpu_compute.c -ldl -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// ============================================================================
// PJRT types
// ============================================================================

typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_Event PJRT_Event;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Executable PJRT_Executable;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
typedef struct PJRT_ExecuteContext PJRT_ExecuteContext;
typedef struct PJRT_Extension_Base PJRT_Extension_Base;
typedef struct PJRT_MultiSlice_Config PJRT_MultiSlice_Config;
typedef struct PJRT_SendCallbackInfo PJRT_SendCallbackInfo;
typedef struct PJRT_RecvCallbackInfo PJRT_RecvCallbackInfo;
typedef struct { size_t struct_size; void* ext; } PJRT_Buffer_MemoryLayout;

#define PJRT_Buffer_Type_F32 11
#define PJRT_HostBufSem_Immutable 0

// ============================================================================
// Arg structs
// ============================================================================

#define HDR  size_t struct_size; void* extension_start

typedef struct { HDR; PJRT_Error* error; } ErrorDestroyArgs;
typedef struct { HDR; const PJRT_Error* error; const char* message; size_t message_size; } ErrorMessageArgs;
typedef struct { HDR; } PluginInitArgs;
typedef struct { HDR; void* a; size_t b; void* c; void* d; void* e; void* f; PJRT_Client* client; void* g; void* h; } ClientCreateArgs;
typedef struct { HDR; PJRT_Client* client; } ClientDestroyArgs;
typedef struct { HDR; PJRT_Client* client; const char* name; size_t name_size; } ClientPlatformNameArgs;
typedef struct { HDR; PJRT_Client* client; PJRT_Device* const* devices; size_t num_devices; } ClientDevicesArgs;

typedef struct {
    HDR;
    PJRT_Client* client;
    const void* data;
    int type;
    const int64_t* dims; size_t num_dims;
    const int64_t* byte_strides; size_t num_byte_strides;
    int host_buffer_semantics;
    PJRT_Device* device;
    PJRT_Memory* memory;
    PJRT_Buffer_MemoryLayout* device_layout;
    PJRT_Event* done_with_host_buffer;  // out
    PJRT_Buffer* buffer;                // out
} BufFromHostArgs;

typedef struct { HDR; PJRT_Event* event; } EventDestroyArgs;
typedef struct { HDR; PJRT_Event* event; } EventAwaitArgs;
typedef struct { HDR; PJRT_Buffer* buffer; } BufDestroyArgs;
typedef struct { HDR; PJRT_Buffer* buffer; const int64_t* dims; size_t num_dims; } BufDimsArgs;
typedef struct { HDR; PJRT_Buffer* buffer; size_t size; } BufOnDevSizeArgs;

typedef struct {
    HDR;
    PJRT_Buffer* src;
    PJRT_Buffer_MemoryLayout* host_layout;
    void* dst;
    size_t dst_size;
    PJRT_Event* event;  // out
} BufToHostArgs;

// Program struct for compile
typedef struct {
    HDR;
    char* code;
    size_t code_size;
    const char* format;
    size_t format_size;
} PjrtProgram;

// Compile args
typedef struct {
    HDR;
    PJRT_Client* client;
    const PjrtProgram* program;
    const char* compile_options;
    size_t compile_options_size;
    PJRT_LoadedExecutable* executable;  // out
} ClientCompileArgs;

// Execute options
typedef struct {
    HDR;
    PJRT_SendCallbackInfo** send_callbacks;
    PJRT_RecvCallbackInfo** recv_callbacks;
    size_t num_send_ops;
    size_t num_recv_ops;
    int launch_id;
    const int64_t* non_donatable_input_indices;
    size_t num_non_donatable_input_indices;
    PJRT_ExecuteContext* context;
    const char* call_location;
    size_t num_tasks;
    int* task_ids;
    int64_t* incarnation_ids;
    PJRT_MultiSlice_Config* multi_slice_config;
} ExecuteOptions;

// Execute args
typedef struct {
    HDR;
    PJRT_LoadedExecutable* executable;
    ExecuteOptions* options;
    PJRT_Buffer* const* const* argument_lists;
    size_t num_devices;
    size_t num_args;
    PJRT_Buffer** const* output_lists;
    PJRT_Event** device_complete_events;
    PJRT_Device* execute_device;
} ExecuteArgs;

// GetExecutable args
typedef struct {
    HDR;
    PJRT_LoadedExecutable* loaded_executable;
    PJRT_Executable* executable;  // out
} GetExecutableArgs;

// NumOutputs args
typedef struct {
    HDR;
    PJRT_Executable* executable;
    size_t num_outputs;  // out
} NumOutputsArgs;

// LoadedExecutable destroy
typedef struct { HDR; PJRT_LoadedExecutable* executable; } LoadedExecDestroyArgs;

// Device description
typedef struct { HDR; PJRT_Device* device; PJRT_DeviceDescription* dd; } DevGetDescArgs;
typedef struct { HDR; PJRT_DeviceDescription* dd; const char* s; size_t s_size; } DevDescToStringArgs;

// ============================================================================
// Function pointer indices (offset from api_ptr + 40)
// ============================================================================

enum {
    FN_ERROR_DESTROY = 0,
    FN_ERROR_MESSAGE = 1,
    FN_PLUGIN_INIT = 3,
    FN_EVENT_DESTROY = 5,
    FN_EVENT_AWAIT = 8,
    FN_CLIENT_CREATE = 10,
    FN_CLIENT_DESTROY = 11,
    FN_CLIENT_PLATFORMNAME = 12,
    FN_CLIENT_DEVICES = 15,
    FN_CLIENT_COMPILE = 20,
    FN_CLIENT_BUFFROMHOST = 22,
    FN_DEVDESC_TOSTRING = 28,
    FN_DEV_GETDESC = 29,
    // Executable ops (from pjrt_c_api.h order)
    FN_EXEC_DESTROY = 40,
    FN_EXEC_NAME = 41,
    FN_EXEC_NUMREPLICAS = 42,
    FN_EXEC_NUMPARTITIONS = 43,
    FN_EXEC_NUMOUTPUTS = 44,
    FN_EXEC_SIZEOFCODE = 45,
    // LoadedExecutable
    FN_LEXEC_DESTROY = 50,
    FN_LEXEC_GETEXEC = 51,
    FN_LEXEC_ADDRDEVICES = 52,
    FN_LEXEC_DELETE = 53,
    FN_LEXEC_ISDELETED = 54,
    FN_LEXEC_EXECUTE = 55,
    // Buffer ops
    FN_BUF_DESTROY = 58,
    FN_BUF_ELEMTYPE = 59,
    FN_BUF_DIMS = 60,
    FN_BUF_ONDEVSZ = 64,
    FN_BUF_TOHOST = 70,
};

static void** fn;

#define CALL(idx, T, a) ((PJRT_Error*(*)(T*))(fn[idx]))(a)
#define CALLV(idx, T, a) ((void(*)(T*))(fn[idx]))(a)

static void check(PJRT_Error* e, const char* ctx) {
    if (!e) return;
    ErrorMessageArgs m = { .struct_size = sizeof(m), .error = e };
    CALLV(FN_ERROR_MESSAGE, ErrorMessageArgs, &m);
    fprintf(stderr, "PJRT Error [%s]: %.*s\n", ctx, (int)m.message_size, m.message);
    ErrorDestroyArgs d = { .struct_size = sizeof(d), .error = e };
    CALLV(FN_ERROR_DESTROY, ErrorDestroyArgs, &d);
    exit(1);
}

static void await_and_destroy_event(PJRT_Event* ev) {
    if (!ev) return;
    EventAwaitArgs a = { .struct_size = sizeof(a), .event = ev };
    check(CALL(FN_EVENT_AWAIT, EventAwaitArgs, &a), "EventAwait");
    EventDestroyArgs d = { .struct_size = sizeof(d), .event = ev };
    CALLV(FN_EVENT_DESTROY, EventDestroyArgs, &d);
}

static PJRT_Buffer* host_to_device(PJRT_Client* client, PJRT_Device* dev,
                                    const float* data, const int64_t* dims,
                                    size_t ndims) {
    BufFromHostArgs a = {
        .struct_size = sizeof(a), .client = client,
        .data = data, .type = PJRT_Buffer_Type_F32,
        .dims = dims, .num_dims = ndims,
        .host_buffer_semantics = PJRT_HostBufSem_Immutable,
        .device = dev,
    };
    check(CALL(FN_CLIENT_BUFFROMHOST, BufFromHostArgs, &a), "BufFromHost");
    await_and_destroy_event(a.done_with_host_buffer);
    return a.buffer;
}

static void device_to_host(PJRT_Buffer* buf, float* dst, size_t nbytes) {
    BufToHostArgs a = { .struct_size = sizeof(a), .src = buf, .dst = dst, .dst_size = nbytes };
    check(CALL(FN_BUF_TOHOST, BufToHostArgs, &a), "BufToHost");
    await_and_destroy_event(a.event);
}

static void destroy_buf(PJRT_Buffer* buf) {
    BufDestroyArgs a = { .struct_size = sizeof(a), .buffer = buf };
    CALLV(FN_BUF_DESTROY, BufDestroyArgs, &a);
}

// Read file into malloc'd buffer
static char* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);
    *out_size = sz;
    return buf;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================\n");
    printf("  TPU Computation: Compile & Execute HLO    \n");
    printf("============================================\n\n");

    // Load library
    const char* lib_path = getenv("LIBTPU_PATH");
    if (!lib_path) lib_path = "/home/kathirks_gc/.local/lib/python3.10/site-packages/libtpu/libtpu.so";
    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    typedef void* (*GetApiFn)(void);
    void* api = ((GetApiFn)dlsym(handle, "GetPjrtApi"))();
    fn = (void**)((char*)api + 40);

    // Init plugin & create client
    printf("[1] Init plugin + create client...\n");
    PluginInitArgs pi = { .struct_size = sizeof(pi) };
    check(CALL(FN_PLUGIN_INIT, PluginInitArgs, &pi), "PluginInit");

    ClientCreateArgs cc = { .struct_size = sizeof(cc) };
    check(CALL(FN_CLIENT_CREATE, ClientCreateArgs, &cc), "ClientCreate");
    PJRT_Client* client = cc.client;

    ClientPlatformNameArgs pn = { .struct_size = sizeof(pn), .client = client };
    check(CALL(FN_CLIENT_PLATFORMNAME, ClientPlatformNameArgs, &pn), "PlatName");
    printf("    Platform: %.*s\n", (int)pn.name_size, pn.name);

    ClientDevicesArgs dv = { .struct_size = sizeof(dv), .client = client };
    check(CALL(FN_CLIENT_DEVICES, ClientDevicesArgs, &dv), "Devices");
    printf("    Devices: %zu\n\n", dv.num_devices);

    PJRT_Device* dev0 = dv.devices[0];

    // Load compile options (num_replicas=1, num_partitions=1)
    size_t compile_opts_size;
    char* compile_opts = read_file("/home/kathirks_gc/tpu_direct/compile_opts.pb", &compile_opts_size);
    printf("    Compile options: %zu bytes\n\n", compile_opts_size);

    // ========================================================================
    // Test 1: Element-wise add (x + x)
    // ========================================================================
    printf("================================================\n");
    printf("  TEST 1: Element-wise add (x + x) on TPU       \n");
    printf("================================================\n\n");

    // Compile HLO
    printf("[2] Compiling add.hlo.pb...\n");
    size_t add_hlo_size;
    char* add_hlo = read_file("/home/kathirks_gc/tpu_direct/add.hlo.pb", &add_hlo_size);
    printf("    HLO size: %zu bytes\n", add_hlo_size);

    PjrtProgram add_prog = {
        .struct_size = sizeof(add_prog),
        .code = add_hlo,
        .code_size = add_hlo_size,
        .format = "hlo",
        .format_size = 3,
    };

    ClientCompileArgs add_comp = {
        .struct_size = sizeof(add_comp),
        .client = client,
        .program = &add_prog,
        .compile_options = compile_opts,
        .compile_options_size = compile_opts_size,
    };
    check(CALL(FN_CLIENT_COMPILE, ClientCompileArgs, &add_comp), "CompileAdd");
    PJRT_LoadedExecutable* add_exec = add_comp.executable;
    printf("    Compiled: %p\n", (void*)add_exec);

    // Get num outputs
    GetExecutableArgs ge = { .struct_size = sizeof(ge), .loaded_executable = add_exec };
    check(CALL(FN_LEXEC_GETEXEC, GetExecutableArgs, &ge), "GetExec");
    NumOutputsArgs no = { .struct_size = sizeof(no), .executable = ge.executable };
    check(CALL(FN_EXEC_NUMOUTPUTS, NumOutputsArgs, &no), "NumOutputs");
    printf("    Outputs per device: %zu\n", no.num_outputs);

    // Prepare input: x = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    float x_data[16];
    for (int i = 0; i < 16; i++) x_data[i] = (float)(i + 1);
    int64_t shape44[] = {4, 4};

    printf("\n[3] Transferring input to TPU...\n");
    printf("    x = [1..16] as 4x4\n");
    PJRT_Buffer* x_buf = host_to_device(client, dev0, x_data, shape44, 2);

    // Execute
    printf("\n[4] Executing x + x on TPU...\n");
    ExecuteOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = sizeof(opts);

    PJRT_Buffer* const arg_list[] = { x_buf };
    PJRT_Buffer* const* arg_lists[] = { arg_list };

    PJRT_Buffer* out_buf = NULL;
    PJRT_Buffer** out_list[] = { &out_buf };

    PJRT_Event* complete_event = NULL;
    PJRT_Event** events[] = { &complete_event };

    ExecuteArgs ex = {
        .struct_size = sizeof(ex),
        .executable = add_exec,
        .options = &opts,
        .argument_lists = arg_lists,
        .num_devices = 1,
        .num_args = 1,
        .output_lists = out_list,
        .device_complete_events = events[0],
        .execute_device = dev0,
    };
    check(CALL(FN_LEXEC_EXECUTE, ExecuteArgs, &ex), "Execute add");

    // Wait for completion
    if (complete_event) await_and_destroy_event(complete_event);
    printf("    Execution complete!\n");

    // Read result
    float add_result[16] = {0};
    device_to_host(out_buf, add_result, sizeof(add_result));

    printf("    Result (x + x):\n    [");
    for (int i = 0; i < 16; i++) {
        printf("%.0f%s", add_result[i], i < 15 ? ", " : "");
        if (i == 7) printf("\n     ");
    }
    printf("]\n");

    // Verify
    int add_ok = 1;
    for (int i = 0; i < 16; i++)
        if (fabsf(add_result[i] - x_data[i] * 2.0f) > 0.001f) add_ok = 0;
    printf("    Verify: %s\n", add_ok ? "PASSED" : "FAILED");

    destroy_buf(out_buf);
    destroy_buf(x_buf);

    // ========================================================================
    // Test 2: Matrix multiply (x @ y)
    // ========================================================================
    printf("\n================================================\n");
    printf("  TEST 2: Matrix multiply (4x4 @ 4x4) on TPU    \n");
    printf("================================================\n\n");

    printf("[5] Compiling matmul.hlo.pb...\n");
    size_t mm_hlo_size;
    char* mm_hlo = read_file("/home/kathirks_gc/tpu_direct/matmul.hlo.pb", &mm_hlo_size);

    PjrtProgram mm_prog = {
        .struct_size = sizeof(mm_prog),
        .code = mm_hlo,
        .code_size = mm_hlo_size,
        .format = "hlo",
        .format_size = 3,
    };
    ClientCompileArgs mm_comp = {
        .struct_size = sizeof(mm_comp),
        .client = client,
        .program = &mm_prog,
        .compile_options = compile_opts,
        .compile_options_size = compile_opts_size,
    };
    check(CALL(FN_CLIENT_COMPILE, ClientCompileArgs, &mm_comp), "CompileMatmul");
    PJRT_LoadedExecutable* mm_exec = mm_comp.executable;
    printf("    Compiled: %p\n", (void*)mm_exec);

    // Inputs: A = identity matrix, B = [[1..16]]
    printf("\n[6] Transferring inputs...\n");
    float a_data[16] = {0};
    a_data[0] = a_data[5] = a_data[10] = a_data[15] = 1.0f;  // identity
    float b_data[16];
    for (int i = 0; i < 16; i++) b_data[i] = (float)(i + 1);

    printf("    A = I(4x4) (identity)\n");
    printf("    B = [1..16] as 4x4\n");

    PJRT_Buffer* a_buf = host_to_device(client, dev0, a_data, shape44, 2);
    PJRT_Buffer* b_buf = host_to_device(client, dev0, b_data, shape44, 2);

    // Execute matmul
    printf("\n[7] Executing A @ B on TPU...\n");
    PJRT_Buffer* const mm_arg_list[] = { a_buf, b_buf };
    PJRT_Buffer* const* mm_arg_lists[] = { mm_arg_list };

    PJRT_Buffer* mm_out = NULL;
    PJRT_Buffer** mm_out_list[] = { &mm_out };

    PJRT_Event* mm_event = NULL;

    ExecuteArgs mm_ex = {
        .struct_size = sizeof(mm_ex),
        .executable = mm_exec,
        .options = &opts,
        .argument_lists = mm_arg_lists,
        .num_devices = 1,
        .num_args = 2,
        .output_lists = mm_out_list,
        .device_complete_events = &mm_event,
        .execute_device = dev0,
    };
    check(CALL(FN_LEXEC_EXECUTE, ExecuteArgs, &mm_ex), "Execute matmul");
    if (mm_event) await_and_destroy_event(mm_event);
    printf("    Execution complete!\n");

    // Read result
    float mm_result[16] = {0};
    device_to_host(mm_out, mm_result, sizeof(mm_result));

    printf("    Result (I @ B = B):\n");
    for (int r = 0; r < 4; r++) {
        printf("    [");
        for (int c = 0; c < 4; c++)
            printf("%6.1f", mm_result[r * 4 + c]);
        printf("  ]\n");
    }

    // Verify: I @ B = B
    int mm_ok = 1;
    for (int i = 0; i < 16; i++)
        if (fabsf(mm_result[i] - b_data[i]) > 0.001f) mm_ok = 0;
    printf("    Verify (I @ B == B): %s\n", mm_ok ? "PASSED" : "FAILED");

    // Now do a real matmul: B @ B
    printf("\n[8] Executing B @ B on TPU...\n");
    PJRT_Buffer* b_buf2 = host_to_device(client, dev0, b_data, shape44, 2);
    PJRT_Buffer* const bb_arg_list[] = { b_buf, b_buf2 };
    PJRT_Buffer* const* bb_arg_lists[] = { bb_arg_list };
    PJRT_Buffer* bb_out = NULL;
    PJRT_Buffer** bb_out_list[] = { &bb_out };
    PJRT_Event* bb_event = NULL;

    ExecuteArgs bb_ex = {
        .struct_size = sizeof(bb_ex),
        .executable = mm_exec,
        .options = &opts,
        .argument_lists = bb_arg_lists,
        .num_devices = 1,
        .num_args = 2,
        .output_lists = bb_out_list,
        .device_complete_events = &bb_event,
        .execute_device = dev0,
    };
    check(CALL(FN_LEXEC_EXECUTE, ExecuteArgs, &bb_ex), "Execute B@B");
    if (bb_event) await_and_destroy_event(bb_event);

    float bb_result[16] = {0};
    device_to_host(bb_out, bb_result, sizeof(bb_result));

    printf("    Result (B @ B):\n");
    for (int r = 0; r < 4; r++) {
        printf("    [");
        for (int c = 0; c < 4; c++)
            printf("%7.0f", bb_result[r * 4 + c]);
        printf("  ]\n");
    }

    // CPU reference for B @ B
    float ref[16] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                ref[i*4+j] += b_data[i*4+k] * b_data[k*4+j];

    printf("    CPU ref (B @ B):\n");
    for (int r = 0; r < 4; r++) {
        printf("    [");
        for (int c = 0; c < 4; c++)
            printf("%7.0f", ref[r * 4 + c]);
        printf("  ]\n");
    }

    int bb_ok = 1;
    for (int i = 0; i < 16; i++)
        if (fabsf(bb_result[i] - ref[i]) > 0.5f) bb_ok = 0;
    printf("    Verify: %s\n", bb_ok ? "PASSED" : "FAILED");

    // Cleanup
    printf("\n[9] Cleanup...\n");
    destroy_buf(mm_out);
    destroy_buf(bb_out);
    destroy_buf(a_buf);
    destroy_buf(b_buf);
    destroy_buf(b_buf2);

    LoadedExecDestroyArgs led1 = { .struct_size = sizeof(led1), .executable = add_exec };
    CALLV(FN_LEXEC_DESTROY, LoadedExecDestroyArgs, &led1);
    LoadedExecDestroyArgs led2 = { .struct_size = sizeof(led2), .executable = mm_exec };
    CALLV(FN_LEXEC_DESTROY, LoadedExecDestroyArgs, &led2);

    ClientDestroyArgs cd = { .struct_size = sizeof(cd), .client = client };
    CALLV(FN_CLIENT_DESTROY, ClientDestroyArgs, &cd);

    free(add_hlo);
    free(mm_hlo);
    free(compile_opts);
    dlclose(handle);
    printf("    Done!\n\n");
    printf("=== Matrix multiply computed on TPU via pure C! ===\n");
    return 0;
}
