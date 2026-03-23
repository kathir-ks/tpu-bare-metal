/*
 * Direct TPU Control via libtpu.so PJRT C API
 * =============================================
 *
 * Uses dlopen/dlsym to load libtpu.so and access the PJRT plugin API.
 * Demonstrates: initialization, device enumeration, host<->TPU data transfer.
 *
 * Compile: gcc -o tpu_pjrt_test tpu_pjrt_test.c -ldl
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdbool.h>

// Opaque PJRT types
typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_Event PJRT_Event;
typedef struct PJRT_Extension_Base PJRT_Extension_Base;

typedef enum {
    PJRT_Buffer_Type_F32 = 11,
} PJRT_Buffer_Type;

typedef enum {
    PJRT_HostBufferSemantics_kImmutableOnlyDuringCall = 0,
} PJRT_HostBufferSemantics;

typedef struct { size_t struct_size; void* ext; /* ... */ } PJRT_Buffer_MemoryLayout;

// ============================================================================
// All arg structs match xla/pjrt/c/pjrt_c_api.h exactly
// Each starts with {size_t struct_size; PJRT_Extension_Base* extension_start;}
// ============================================================================

#define PJRT_ARGS_HEADER  size_t struct_size; void* extension_start

typedef struct { PJRT_ARGS_HEADER; PJRT_Error* error; } ErrorDestroyArgs;
typedef struct { PJRT_ARGS_HEADER; const PJRT_Error* error; const char* message; size_t message_size; } ErrorMessageArgs;
typedef struct { PJRT_ARGS_HEADER; } PluginInitArgs;
typedef struct { PJRT_ARGS_HEADER; void* attributes; size_t num_attributes; } PluginAttrsArgs;

typedef struct { PJRT_ARGS_HEADER; PJRT_Event* event; } EventDestroyArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Event* event; } EventAwaitArgs;

typedef struct {
    PJRT_ARGS_HEADER;
    void* create_options; size_t num_options;
    void* kv_get_cb; void* kv_get_arg;
    void* kv_put_cb; void* kv_put_arg;
    PJRT_Client* client;  // out
    void* kv_try_get_cb; void* kv_try_get_arg;
} ClientCreateArgs;

typedef struct { PJRT_ARGS_HEADER; PJRT_Client* client; } ClientDestroyArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Client* client; const char* platform_name; size_t platform_name_size; } ClientPlatformNameArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Client* client; int process_index; } ClientProcessIndexArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Client* client; const char* platform_version; size_t platform_version_size; } ClientPlatformVersionArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Client* client; PJRT_Device* const* devices; size_t num_devices; } ClientDevicesArgs;

typedef struct {
    PJRT_ARGS_HEADER;
    PJRT_Client* client;
    const void* data;
    int type;  // PJRT_Buffer_Type enum (int-sized)
    const int64_t* dims; size_t num_dims;
    const int64_t* byte_strides; size_t num_byte_strides;
    int host_buffer_semantics;
    PJRT_Device* device;
    PJRT_Memory* memory;
    PJRT_Buffer_MemoryLayout* device_layout;
    PJRT_Event* done_with_host_buffer;  // out
    PJRT_Buffer* buffer;                // out
} ClientBufferFromHostArgs;

typedef struct { PJRT_ARGS_HEADER; PJRT_DeviceDescription* dd; int64_t id; } DevDescIdArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_DeviceDescription* dd; const char* kind; size_t kind_size; } DevDescKindArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_DeviceDescription* dd; const char* s; size_t s_size; } DevDescToStringArgs;

typedef struct { PJRT_ARGS_HEADER; PJRT_Device* device; PJRT_DeviceDescription* dd; } DevGetDescArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Device* device; bool is_addressable; } DevIsAddrArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Device* device; int local_hw_id; } DevLocalHwIdArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Device* device; PJRT_Memory* default_memory; } DevDefaultMemArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Memory* memory; const char* kind; size_t kind_size; } MemKindArgs;

typedef struct { PJRT_ARGS_HEADER; PJRT_Buffer* buffer; } BufDestroyArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Buffer* buffer; int type; } BufElemTypeArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Buffer* buffer; const int64_t* dims; size_t num_dims; } BufDimsArgs;
typedef struct { PJRT_ARGS_HEADER; PJRT_Buffer* buffer; size_t size; } BufOnDevSizeArgs;

typedef struct {
    PJRT_ARGS_HEADER;
    PJRT_Buffer* src;
    PJRT_Buffer_MemoryLayout* host_layout;
    void* dst;
    size_t dst_size;
    PJRT_Event* event;  // out
} BufToHostArgs;

// ============================================================================
// Function pointer indices into the PJRT_Api table
// The table starts at api_ptr + 40 (after 8+8+24 byte header)
// ============================================================================

#define FN_ERROR_DESTROY       0
#define FN_ERROR_MESSAGE       1
#define FN_ERROR_GETCODE       2
#define FN_PLUGIN_INITIALIZE   3
#define FN_PLUGIN_ATTRIBUTES   4
#define FN_EVENT_DESTROY       5
#define FN_EVENT_ISREADY       6
#define FN_EVENT_ERROR         7
#define FN_EVENT_AWAIT         8
#define FN_EVENT_ONREADY       9
#define FN_CLIENT_CREATE       10
#define FN_CLIENT_DESTROY      11
#define FN_CLIENT_PLATFORMNAME 12
#define FN_CLIENT_PROCESSINDEX 13
#define FN_CLIENT_PLATFORMVER  14
#define FN_CLIENT_DEVICES      15
#define FN_CLIENT_ADDRDEVICES  16
#define FN_CLIENT_LOOKUPDEV    17
#define FN_CLIENT_LOOKUPADDR   18
#define FN_CLIENT_ADDRMEM      19
#define FN_CLIENT_COMPILE      20
#define FN_CLIENT_DEFDEVASSIGN 21
#define FN_CLIENT_BUFFROMHOST  22
#define FN_DEVDESC_ID          23
#define FN_DEVDESC_PROCIDX     24
#define FN_DEVDESC_ATTRS       25
#define FN_DEVDESC_KIND        26
#define FN_DEVDESC_DEBUGSTR    27
#define FN_DEVDESC_TOSTRING    28
#define FN_DEV_GETDESC         29
#define FN_DEV_ISADDR          30
#define FN_DEV_LOCALHWID       31
#define FN_DEV_ADDRMEM         32
#define FN_DEV_DEFAULTMEM      33
#define FN_DEV_MEMSTATS        34
#define FN_MEM_ID              35
#define FN_MEM_KIND            36
// ... buffer ops continue further
// Executable ops: 40-49, LoadedExecutable: 50-57
#define FN_BUF_DESTROY         58
#define FN_BUF_ELEMTYPE        59
#define FN_BUF_DIMS            60
#define FN_BUF_UNPADDED        61
#define FN_BUF_DYNIDX          62
#define FN_BUF_LAYOUT          63
#define FN_BUF_ONDEVSZ         64
#define FN_BUF_DEVICE          65
#define FN_BUF_MEMORY          66
#define FN_BUF_DELETE          67
#define FN_BUF_ISDELETED       68
#define FN_BUF_COPYTODEV       69
#define FN_BUF_TOHOST          70

// ============================================================================
// API accessor
// ============================================================================

static void** fn_table = NULL;
static void* api_ptr = NULL;

#define CALL(idx, args_type, args_ptr) \
    ((PJRT_Error* (*)(args_type*))(fn_table[idx]))(args_ptr)

#define CALLV(idx, args_type, args_ptr) \
    ((void (*)(args_type*))(fn_table[idx]))(args_ptr)

static void check_error(PJRT_Error* err, const char* ctx) {
    if (!err) return;
    ErrorMessageArgs ma = { .struct_size = sizeof(ma), .error = err };
    CALLV(FN_ERROR_MESSAGE, ErrorMessageArgs, &ma);
    fprintf(stderr, "ERROR [%s]: %.*s\n", ctx, (int)ma.message_size, ma.message);
    ErrorDestroyArgs da = { .struct_size = sizeof(da), .error = err };
    CALLV(FN_ERROR_DESTROY, ErrorDestroyArgs, &da);
    exit(1);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("============================================\n");
    printf("  Direct TPU Control via libtpu.so (PJRT)   \n");
    printf("============================================\n\n");

    // --- Load library ---
    const char* lib_path = getenv("LIBTPU_PATH");
    if (!lib_path) lib_path = "/home/kathirks_gc/.local/lib/python3.10/site-packages/libtpu/libtpu.so";
    printf("[1] Loading libtpu.so...\n");
    void* handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    typedef void* (*GetPjrtApiFn)(void);
    GetPjrtApiFn get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
    if (!get_api) { fprintf(stderr, "GetPjrtApi not found\n"); return 1; }

    api_ptr = get_api();
    // Header: struct_size(8) + extension_start(8) + PJRT_Api_Version(24) = 40 bytes
    fn_table = (void**)((char*)api_ptr + 40);

    size_t struct_size = *(size_t*)api_ptr;
    int major = *(int*)((char*)api_ptr + 32);
    int minor = *(int*)((char*)api_ptr + 36);
    printf("    PJRT API v%d.%d, %zu function pointers\n\n",
           major, minor, (struct_size - 40) / 8);

    // --- Initialize plugin ---
    printf("[2] Initializing TPU plugin...\n");
    PluginInitArgs pi = { .struct_size = sizeof(pi) };
    check_error(CALL(FN_PLUGIN_INITIALIZE, PluginInitArgs, &pi), "PluginInit");
    printf("    OK\n\n");

    // --- Create client ---
    printf("[3] Creating TPU client...\n");
    ClientCreateArgs cc = { .struct_size = sizeof(cc) };
    check_error(CALL(FN_CLIENT_CREATE, ClientCreateArgs, &cc), "ClientCreate");
    PJRT_Client* client = cc.client;
    printf("    Client: %p\n\n", (void*)client);

    // --- Platform info ---
    printf("[4] Platform info:\n");
    {
        ClientPlatformNameArgs a = { .struct_size = sizeof(a), .client = client };
        check_error(CALL(FN_CLIENT_PLATFORMNAME, ClientPlatformNameArgs, &a), "PlatformName");
        printf("    Platform:  %.*s\n", (int)a.platform_name_size, a.platform_name);
    }
    {
        ClientPlatformVersionArgs a = { .struct_size = sizeof(a), .client = client };
        check_error(CALL(FN_CLIENT_PLATFORMVER, ClientPlatformVersionArgs, &a), "PlatformVer");
        printf("    Version:   %.*s\n", (int)a.platform_version_size, a.platform_version);
    }
    {
        ClientProcessIndexArgs a = { .struct_size = sizeof(a), .client = client };
        check_error(CALL(FN_CLIENT_PROCESSINDEX, ClientProcessIndexArgs, &a), "ProcessIdx");
        printf("    Process:   %d\n", a.process_index);
    }

    // --- Enumerate devices ---
    printf("\n[5] TPU Devices:\n");
    ClientDevicesArgs da = { .struct_size = sizeof(da), .client = client };
    check_error(CALL(FN_CLIENT_DEVICES, ClientDevicesArgs, &da), "Devices");
    printf("    Found %zu devices\n", da.num_devices);

    for (size_t i = 0; i < da.num_devices; i++) {
        PJRT_Device* dev = da.devices[i];

        DevGetDescArgs gd = { .struct_size = sizeof(gd), .device = dev };
        check_error(CALL(FN_DEV_GETDESC, DevGetDescArgs, &gd), "GetDesc");

        DevDescIdArgs id = { .struct_size = sizeof(id), .dd = gd.dd };
        check_error(CALL(FN_DEVDESC_ID, DevDescIdArgs, &id), "DevId");

        DevDescKindArgs kd = { .struct_size = sizeof(kd), .dd = gd.dd };
        check_error(CALL(FN_DEVDESC_KIND, DevDescKindArgs, &kd), "DevKind");

        DevDescToStringArgs ts = { .struct_size = sizeof(ts), .dd = gd.dd };
        check_error(CALL(FN_DEVDESC_TOSTRING, DevDescToStringArgs, &ts), "DevStr");

        DevIsAddrArgs ia = { .struct_size = sizeof(ia), .device = dev };
        check_error(CALL(FN_DEV_ISADDR, DevIsAddrArgs, &ia), "IsAddr");

        DevLocalHwIdArgs hw = { .struct_size = sizeof(hw), .device = dev };
        check_error(CALL(FN_DEV_LOCALHWID, DevLocalHwIdArgs, &hw), "HwId");

        printf("\n    Device %zu:  ID=%ld  Kind=%.*s  HW=%d  Addressable=%s\n",
               i, (long)id.id, (int)kd.kind_size, kd.kind,
               hw.local_hw_id, ia.is_addressable ? "yes" : "no");
        printf("              %.*s\n", (int)ts.s_size, ts.s);

        // Default memory
        DevDefaultMemArgs dm = { .struct_size = sizeof(dm), .device = dev };
        PJRT_Error* merr = CALL(FN_DEV_DEFAULTMEM, DevDefaultMemArgs, &dm);
        if (!merr && dm.default_memory) {
            MemKindArgs mk = { .struct_size = sizeof(mk), .memory = dm.default_memory };
            PJRT_Error* mkerr = CALL(FN_MEM_KIND, MemKindArgs, &mk);
            if (!mkerr)
                printf("              Memory: %.*s\n", (int)mk.kind_size, mk.kind);
            else { ErrorDestroyArgs d = { .struct_size = sizeof(d), .error = mkerr }; CALLV(FN_ERROR_DESTROY, ErrorDestroyArgs, &d); }
        } else if (merr) {
            ErrorDestroyArgs d = { .struct_size = sizeof(d), .error = merr };
            CALLV(FN_ERROR_DESTROY, ErrorDestroyArgs, &d);
        }
    }

    // --- Host -> TPU transfer ---
    printf("\n[6] Host -> TPU data transfer:\n");
    float host_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    int64_t dims[] = {2, 4};
    printf("    Sending float32[2,4] = {1,2,3,4,5,6,7,8}\n");

    ClientBufferFromHostArgs bh = {
        .struct_size = sizeof(bh),
        .client = client,
        .data = host_data,
        .type = PJRT_Buffer_Type_F32,
        .dims = dims,
        .num_dims = 2,
        .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
        .device = da.devices[0],
    };
    check_error(CALL(FN_CLIENT_BUFFROMHOST, ClientBufferFromHostArgs, &bh), "BufFromHost");
    PJRT_Buffer* buf = bh.buffer;
    printf("    Buffer on TPU: %p\n", (void*)buf);

    // Wait for transfer
    if (bh.done_with_host_buffer) {
        EventAwaitArgs aw = { .struct_size = sizeof(aw), .event = bh.done_with_host_buffer };
        check_error(CALL(FN_EVENT_AWAIT, EventAwaitArgs, &aw), "AwaitXfer");
        EventDestroyArgs ed = { .struct_size = sizeof(ed), .event = bh.done_with_host_buffer };
        CALLV(FN_EVENT_DESTROY, EventDestroyArgs, &ed);
    }

    // Buffer properties
    BufOnDevSizeArgs bs = { .struct_size = sizeof(bs), .buffer = buf };
    check_error(CALL(FN_BUF_ONDEVSZ, BufOnDevSizeArgs, &bs), "BufSize");
    printf("    On-device size: %zu bytes\n", bs.size);

    BufDimsArgs bd = { .struct_size = sizeof(bd), .buffer = buf };
    check_error(CALL(FN_BUF_DIMS, BufDimsArgs, &bd), "BufDims");
    printf("    Shape: [");
    for (size_t i = 0; i < bd.num_dims; i++)
        printf("%ld%s", (long)bd.dims[i], i+1 < bd.num_dims ? "," : "");
    printf("]\n");

    BufElemTypeArgs bt = { .struct_size = sizeof(bt), .buffer = buf };
    check_error(CALL(FN_BUF_ELEMTYPE, BufElemTypeArgs, &bt), "BufType");
    printf("    Type: %d (F32=11)\n", bt.type);

    // --- TPU -> Host readback ---
    printf("\n[7] TPU -> Host readback:\n");
    float result[8] = {0};
    BufToHostArgs th = {
        .struct_size = sizeof(th),
        .src = buf,
        .dst = result,
        .dst_size = sizeof(result),
    };
    check_error(CALL(FN_BUF_TOHOST, BufToHostArgs, &th), "BufToHost");

    if (th.event) {
        EventAwaitArgs aw = { .struct_size = sizeof(aw), .event = th.event };
        check_error(CALL(FN_EVENT_AWAIT, EventAwaitArgs, &aw), "AwaitRead");
        EventDestroyArgs ed = { .struct_size = sizeof(ed), .event = th.event };
        CALLV(FN_EVENT_DESTROY, EventDestroyArgs, &ed);
    }

    printf("    Data: [");
    for (int i = 0; i < 8; i++) printf("%.1f%s", result[i], i < 7 ? ", " : "");
    printf("]\n");

    int ok = 1;
    for (int i = 0; i < 8; i++) if (result[i] != host_data[i]) ok = 0;
    printf("    Roundtrip: %s\n", ok ? "PASSED" : "FAILED");

    // --- Cleanup ---
    printf("\n[8] Cleanup...\n");
    BufDestroyArgs bdd = { .struct_size = sizeof(bdd), .buffer = buf };
    CALLV(FN_BUF_DESTROY, BufDestroyArgs, &bdd);

    ClientDestroyArgs cdd = { .struct_size = sizeof(cdd), .client = client };
    CALLV(FN_CLIENT_DESTROY, ClientDestroyArgs, &cdd);

    dlclose(handle);
    printf("    Done!\n\n");
    printf("=== TPU controlled directly from C — no Python/JAX needed ===\n");
    return 0;
}
