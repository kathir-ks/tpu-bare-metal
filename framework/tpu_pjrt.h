/*
 * tpu_pjrt.h — Internal PJRT C API types, function indices, and arg structs.
 *
 * Do NOT include this in user code. Use tpu.h instead.
 *
 * Function pointer indices are 0-based from the start of the function pointer
 * table, which begins at byte offset 40 of PJRT_Api:
 *   +0:  struct_size (8 bytes)
 *   +8:  extension_start (8 bytes)
 *   +16: PJRT_Api_Version { struct_size(8), ext(8), major(4), minor(4) }
 *   +40: fn[0] = PJRT_Error_Destroy, fn[1] = PJRT_Error_Message, ...
 *
 * Verified against:
 *   xla/pjrt/c/pjrt_c_api.h (TF include, PJRT API v0.69)
 */

#ifndef TPU_PJRT_H
#define TPU_PJRT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ── Opaque PJRT handle types ─────────────────────────────────────────────── */

typedef struct PJRT_Error                              PJRT_Error;
typedef struct PJRT_Client                             PJRT_Client;
typedef struct PJRT_Device                             PJRT_Device;
typedef struct PJRT_Memory                             PJRT_Memory;
typedef struct PJRT_Buffer                             PJRT_Buffer;
typedef struct PJRT_Event                              PJRT_Event;
typedef struct PJRT_LoadedExecutable                   PJRT_LoadedExecutable;
typedef struct PJRT_Executable                         PJRT_Executable;
typedef struct PJRT_SerializedExecutable               PJRT_SerializedExecutable;
typedef struct PJRT_DeviceDescription                  PJRT_DeviceDescription;
typedef struct PJRT_TopologyDescription                PJRT_TopologyDescription;
typedef struct PJRT_ExecuteContext                     PJRT_ExecuteContext;
typedef struct PJRT_Extension_Base                     PJRT_Extension_Base;
typedef struct PJRT_SendCallbackInfo                   PJRT_SendCallbackInfo;
typedef struct PJRT_RecvCallbackInfo                   PJRT_RecvCallbackInfo;
typedef struct PJRT_MultiSlice_Config                  PJRT_MultiSlice_Config;
/* ── Common struct header ─────────────────────────────────────────────────── */
/* Every PJRT arg struct starts with these two fields. Always set struct_size. */

#define PJRT_HDR  size_t struct_size; PJRT_Extension_Base* extension_start

/* ── Buffer memory layout (for ToHostBuffer host_layout) ──────────────────── */
/* Mirrors pjrt_c_api.h. Passing an explicit row-major Tiled layout (no tiles)
 * forces a dense host buffer; with NULL host_layout, libtpu may return the
 * device's tiled/padded layout for shapes whose minor dim isn't 128-aligned. */

typedef enum {
    PJRT_Buffer_MemoryLayout_Type_Tiled   = 0,
    PJRT_Buffer_MemoryLayout_Type_Strides = 1,
} PJRT_Buffer_MemoryLayout_Type;

typedef struct {
    PJRT_HDR;
    const int64_t* minor_to_major;
    size_t         minor_to_major_size;
    const int64_t* tile_dims;
    const size_t*  tile_dim_sizes;
    size_t         num_tiles;
} PJRT_Buffer_MemoryLayout_Tiled;

typedef struct {
    PJRT_HDR;
    const int64_t* byte_strides;
    size_t         num_byte_strides;
} PJRT_Buffer_MemoryLayout_Strides;

typedef struct PJRT_Buffer_MemoryLayout {
    PJRT_HDR;
    union {
        PJRT_Buffer_MemoryLayout_Tiled   tiled;
        PJRT_Buffer_MemoryLayout_Strides strides;
    };
    PJRT_Buffer_MemoryLayout_Type type;
} PJRT_Buffer_MemoryLayout;

/* ── Buffer element types (PJRT_Buffer_Type enum) ────────────────────────── */

typedef enum {
    PJRT_Buffer_Type_INVALID = 0,
    PJRT_Buffer_Type_PRED    = 1,
    PJRT_Buffer_Type_S8      = 2,
    PJRT_Buffer_Type_S16     = 3,
    PJRT_Buffer_Type_S32     = 4,
    PJRT_Buffer_Type_S64     = 5,
    PJRT_Buffer_Type_U8      = 6,
    PJRT_Buffer_Type_U16     = 7,
    PJRT_Buffer_Type_U32     = 8,
    PJRT_Buffer_Type_U64     = 9,
    PJRT_Buffer_Type_F16     = 10,
    PJRT_Buffer_Type_F32     = 11,
    PJRT_Buffer_Type_F64     = 12,
    PJRT_Buffer_Type_BF16    = 13,
    PJRT_Buffer_Type_C64     = 14,
    PJRT_Buffer_Type_C128    = 15,
} PJRT_Buffer_Type;

/* ── Host buffer semantics ────────────────────────────────────────────────── */

typedef enum {
    PJRT_HostBuf_ImmutableOnlyDuringCall        = 0,
    PJRT_HostBuf_ImmutableUntilTransferCompletes = 1,
    PJRT_HostBuf_ImmutableZeroCopy              = 2,
    PJRT_HostBuf_MutableZeroCopy                = 3,
} PJRT_HostBufferSemantics;

/* ── Function table indices ───────────────────────────────────────────────── */
/* 0-indexed from fn[0] = first function pointer at api_ptr+40.               */

enum {
    /* Error handling */
    FN_ERROR_DESTROY          = 0,
    FN_ERROR_MESSAGE          = 1,
    FN_ERROR_GETCODE          = 2,

    /* Plugin */
    FN_PLUGIN_INITIALIZE      = 3,
    FN_PLUGIN_ATTRIBUTES      = 4,

    /* Events */
    FN_EVENT_DESTROY          = 5,
    FN_EVENT_ISREADY          = 6,
    FN_EVENT_ERROR            = 7,
    FN_EVENT_AWAIT            = 8,
    FN_EVENT_ONREADY          = 9,

    /* Client lifecycle */
    FN_CLIENT_CREATE          = 10,
    FN_CLIENT_DESTROY         = 11,
    FN_CLIENT_PLATFORMNAME    = 12,
    FN_CLIENT_PROCESSINDEX    = 13,
    FN_CLIENT_PLATFORMVERSION = 14,
    FN_CLIENT_DEVICES         = 15,
    FN_CLIENT_ADDRDEVICES     = 16,
    FN_CLIENT_LOOKUPDEV       = 17,
    FN_CLIENT_LOOKUPADDRDEV   = 18,
    FN_CLIENT_ADDRMEMS        = 19,
    FN_CLIENT_COMPILE         = 20,
    FN_CLIENT_DEFDEVASSIGN    = 21,
    FN_CLIENT_BUFFROMHOST     = 22,

    /* Device description */
    FN_DEVDESC_ID             = 23,
    FN_DEVDESC_PROCIDX        = 24,
    FN_DEVDESC_ATTRS          = 25,
    FN_DEVDESC_KIND           = 26,
    FN_DEVDESC_DEBUGSTR       = 27,
    FN_DEVDESC_TOSTRING       = 28,

    /* Device */
    FN_DEV_GETDESC            = 29,
    FN_DEV_ISADDR             = 30,
    FN_DEV_LOCALHWID          = 31,
    FN_DEV_ADDRMEMS           = 32,
    FN_DEV_DEFAULTMEM         = 33,
    FN_DEV_MEMSTATS           = 34,

    /* Memory */
    FN_MEM_ID                 = 35,
    FN_MEM_KIND               = 36,
    FN_MEM_DEBUGSTR           = 37,
    FN_MEM_TOSTRING           = 38,
    FN_MEM_ADDRBYDEVS         = 39,

    /* Executable (unloaded) */
    FN_EXEC_DESTROY           = 40,
    FN_EXEC_NAME              = 41,
    FN_EXEC_NUMREPLICAS       = 42,
    FN_EXEC_NUMPARTITIONS     = 43,
    FN_EXEC_NUMOUTPUTS        = 44,
    FN_EXEC_SIZEOFCODE        = 45,
    FN_EXEC_COSTANALYSIS      = 46,
    FN_EXEC_OUTPUTMEMKINDS    = 47,
    FN_EXEC_OPTIMIZEDPROG     = 48,
    FN_EXEC_SERIALIZE         = 49,  /* compile-once save */

    /* Loaded executable */
    FN_LEXEC_DESTROY          = 50,
    FN_LEXEC_GETEXEC          = 51,
    FN_LEXEC_ADDRDEVICES      = 52,
    FN_LEXEC_DELETE           = 53,
    FN_LEXEC_ISDELETED        = 54,
    FN_LEXEC_EXECUTE          = 55,
    FN_EXEC_DESERIALIZE_LOAD  = 56,  /* load saved executable */
    FN_LEXEC_FINGERPRINT      = 57,

    /* Buffer */
    FN_BUF_DESTROY            = 58,
    FN_BUF_ELEMTYPE           = 59,
    FN_BUF_DIMS               = 60,
    FN_BUF_UNPADDEDDIMS       = 61,
    FN_BUF_DYNIDX             = 62,
    FN_BUF_LAYOUT             = 63,
    FN_BUF_ONDEVSZ            = 64,
    FN_BUF_DEVICE             = 65,
    FN_BUF_MEMORY             = 66,
    FN_BUF_DELETE             = 67,
    FN_BUF_ISDELETED          = 68,
    FN_BUF_COPYTODEV          = 69,  /* device-to-device, no host roundtrip */
    FN_BUF_TOHOST             = 70,
    FN_BUF_ISONCPU            = 71,
    FN_BUF_READYEVENT         = 72,
    FN_BUF_UNSAFEPTR          = 73,

    /* Topology */
    FN_CLIENT_TOPODESC        = 95,
};

/* ── Call macros ──────────────────────────────────────────────────────────── */
/* fn_table is void** pointing to api_ptr+40. Passed explicitly to support     */
/* multiple concurrent contexts without globals.                               */

#define PJRT_CALL(fn_table, idx, T, a) \
    ((PJRT_Error*(*)(T*))(fn_table[idx]))(a)
#define PJRT_CALLV(fn_table, idx, T, a) \
    ((void(*)(T*))(fn_table[idx]))(a)

/* ══════════════════════════════════════════════════════════════════════════ */
/* Arg structs — ordered by function index.                                   */
/* Each field after the HDR is named exactly as in pjrt_c_api.h.              */
/* ══════════════════════════════════════════════════════════════════════════ */

/* fn[0] */
typedef struct { PJRT_HDR; PJRT_Error* error; } PJRT_Error_Destroy_Args;
/* fn[1] — message / message_size are OUT */
typedef struct {
    PJRT_HDR;
    const PJRT_Error* error;
    const char*       message;       /* out */
    size_t            message_size;  /* out */
} PJRT_Error_Message_Args;

/* fn[3] */
typedef struct { PJRT_HDR; } PJRT_Plugin_Initialize_Args;

/* fn[5] */
typedef struct { PJRT_HDR; PJRT_Event* event; } PJRT_Event_Destroy_Args;
/* fn[6] — is_ready is OUT */
typedef struct { PJRT_HDR; PJRT_Event* event; bool is_ready; } PJRT_Event_IsReady_Args;
/* fn[8] */
typedef struct { PJRT_HDR; PJRT_Event* event; } PJRT_Event_Await_Args;

/* fn[9] — async callback */
typedef void (*PJRT_Event_OnReadyCallback)(PJRT_Error* error, void* user_arg);
typedef struct {
    PJRT_HDR;
    PJRT_Event*                event;
    PJRT_Event_OnReadyCallback callback;
    void*                      user_arg;
} PJRT_Event_OnReady_Args;

/* fn[10] — client is OUT */
typedef struct {
    PJRT_HDR;
    void*        create_options;
    size_t       num_options;
    void*        kv_get_callback;
    void*        kv_get_user_arg;
    void*        kv_put_callback;
    void*        kv_put_user_arg;
    PJRT_Client* client;           /* out */
    void*        kv_try_get_callback;
    void*        kv_try_get_user_arg;
} PJRT_Client_Create_Args;

/* fn[11] */
typedef struct { PJRT_HDR; PJRT_Client* client; } PJRT_Client_Destroy_Args;

/* fn[12] — platform_name / platform_name_size are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Client* client;
    const char*  platform_name;       /* out */
    size_t       platform_name_size;  /* out */
} PJRT_Client_PlatformName_Args;

/* fn[13] — process_index is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Client* client;
    int          process_index;  /* out */
} PJRT_Client_ProcessIndex_Args;

/* fn[14] — platform_version / platform_version_size are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Client* client;
    const char*  platform_version;       /* out */
    size_t       platform_version_size;  /* out */
} PJRT_Client_PlatformVersion_Args;

/* fn[15] — devices / num_devices are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Client*         client;
    PJRT_Device* const*  devices;      /* out */
    size_t               num_devices;  /* out */
} PJRT_Client_Devices_Args;

/* fn[16] — same shape as Devices but for addressable only */
typedef struct {
    PJRT_HDR;
    PJRT_Client*         client;
    PJRT_Device* const*  addressable_devices;      /* out */
    size_t               num_addressable_devices;  /* out */
} PJRT_Client_AddressableDevices_Args;

/* fn[20] — executable is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Client*            client;
    const void*             program;                 /* ptr to PJRT_Program */
    const char*             compile_options;
    size_t                  compile_options_size;
    PJRT_LoadedExecutable*  executable;  /* out */
} PJRT_Client_Compile_Args;

/* PJRT_Program — passed by pointer in compile_options field above */
typedef struct {
    PJRT_HDR;
    char*       code;
    size_t      code_size;
    const char* format;
    size_t      format_size;
} PJRT_Program;

/* fn[22] — done_with_host_buffer and buffer are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Client*              client;
    const void*               data;
    int                       type;                    /* PJRT_Buffer_Type */
    const int64_t*            dims;
    size_t                    num_dims;
    const int64_t*            byte_strides;
    size_t                    num_byte_strides;
    int                       host_buffer_semantics;   /* PJRT_HostBufferSemantics */
    PJRT_Device*              device;
    PJRT_Memory*              memory;
    PJRT_Buffer_MemoryLayout* device_layout;
    PJRT_Event*               done_with_host_buffer;   /* out — may be NULL */
    PJRT_Buffer*              buffer;                  /* out */
} PJRT_Client_BufferFromHostBuffer_Args;

/* fn[23] — id is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_DeviceDescription* device_description;
    int64_t                 id;  /* out */
} PJRT_DeviceDescription_Id_Args;

/* fn[26] — kind / kind_size are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_DeviceDescription* device_description;
    const char*             device_kind;       /* out */
    size_t                  device_kind_size;  /* out */
} PJRT_DeviceDescription_Kind_Args;

/* fn[28] — to_string / to_string_size are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_DeviceDescription* device_description;
    const char*             to_string;       /* out */
    size_t                  to_string_size;  /* out */
} PJRT_DeviceDescription_ToString_Args;

/* fn[29] — device_description is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Device*             device;
    PJRT_DeviceDescription*  device_description;  /* out */
} PJRT_Device_GetDescription_Args;

/* fn[30] — is_addressable is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Device* device;
    bool         is_addressable;  /* out */
} PJRT_Device_IsAddressable_Args;

/* fn[31] — local_hardware_id is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Device* device;
    int          local_hardware_id;  /* out */
} PJRT_Device_LocalHardwareId_Args;

/* fn[33] — default_memory is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Device* device;
    PJRT_Memory* default_memory;  /* out */
} PJRT_Device_DefaultMemory_Args;

/* fn[34] — bytes_in_use / bytes_limit are OUT; bytes_limit_is_set indicates validity */
typedef struct {
    PJRT_HDR;
    PJRT_Device* device;
    int64_t      bytes_in_use;      /* out */
    int64_t      bytes_limit;       /* out */
    bool         bytes_limit_is_set; /* out */
} PJRT_Device_MemoryStats_Args;

/* fn[36] — kind / kind_size are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Memory* memory;
    const char*  kind;       /* out */
    size_t       kind_size;  /* out */
} PJRT_Memory_Kind_Args;

/* fn[44] — num_outputs is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Executable* executable;
    size_t           num_outputs;  /* out */
} PJRT_Executable_NumOutputs_Args;

/* fn[49] — serialized_bytes, serialized_bytes_size, serialized_executable,
            serialized_executable_deleter are OUT */
typedef struct {
    PJRT_HDR;
    const PJRT_Executable*   executable;
    const char*              serialized_bytes;        /* out */
    size_t                   serialized_bytes_size;   /* out */
    PJRT_SerializedExecutable* serialized_executable; /* out */
    void (*serialized_executable_deleter)(PJRT_SerializedExecutable*);  /* out */
} PJRT_Executable_Serialize_Args;

/* fn[50] */
typedef struct {
    PJRT_HDR;
    PJRT_LoadedExecutable* executable;
} PJRT_LoadedExecutable_Destroy_Args;

/* fn[51] — executable is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_LoadedExecutable* loaded_executable;
    PJRT_Executable*       executable;  /* out */
} PJRT_LoadedExecutable_GetExecutable_Args;

/* fn[52] — devices in replica order (caller must NOT free) */
typedef struct {
    PJRT_HDR;
    PJRT_LoadedExecutable*  executable;
    PJRT_Device* const*     addressable_devices;  /* out */
    size_t                  num_addressable_devices; /* out */
} PJRT_LoadedExecutable_AddressableDevices_Args;

/* fn[55] — output_lists and device_complete_events are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_SendCallbackInfo**           send_callbacks;
    PJRT_RecvCallbackInfo**           recv_callbacks;
    size_t                            num_send_ops;
    size_t                            num_recv_ops;
    int                               launch_id;
    const int64_t*                    non_donatable_input_indices;
    size_t                            num_non_donatable_input_indices;
    PJRT_ExecuteContext*              context;
    const char*                       call_location;
    size_t                            num_tasks;
    int*                              task_ids;
    int64_t*                          incarnation_ids;
    PJRT_MultiSlice_Config*           multi_slice_config;
} PJRT_ExecuteOptions;

typedef struct {
    PJRT_HDR;
    PJRT_LoadedExecutable*      executable;
    PJRT_ExecuteOptions*        options;
    PJRT_Buffer* const* const*  argument_lists;    /* [num_devices][num_args] */
    size_t                      num_devices;
    size_t                      num_args;
    PJRT_Buffer** const*        output_lists;      /* [num_devices][num_outputs], out */
    PJRT_Event**                device_complete_events; /* [num_devices], out */
    PJRT_Device*                execute_device;    /* NULL = all devices */
} PJRT_LoadedExecutable_Execute_Args;

/* fn[56] — loaded_executable is OUT (comes before optional override fields) */
typedef struct {
    PJRT_HDR;
    PJRT_Client*             client;
    const char*              serialized_executable;
    size_t                   serialized_executable_size;
    PJRT_LoadedExecutable*   loaded_executable;                         /* out */
    const char*              overridden_serialized_compile_options;     /* may be NULL */
    size_t                   overridden_serialized_compile_options_size;
} PJRT_Executable_DeserializeAndLoad_Args;

/* fn[58] */
typedef struct { PJRT_HDR; PJRT_Buffer* buffer; } PJRT_Buffer_Destroy_Args;

/* fn[59] — element_type is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Buffer* buffer;
    int          element_type;  /* PJRT_Buffer_Type, out */
} PJRT_Buffer_ElementType_Args;

/* fn[60] — dims / num_dims are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Buffer*   buffer;
    const int64_t* dims;      /* out */
    size_t         num_dims;  /* out */
} PJRT_Buffer_Dimensions_Args;

/* fn[64] — on_device_size_in_bytes is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Buffer* buffer;
    size_t       on_device_size_in_bytes;  /* out */
} PJRT_Buffer_OnDeviceSizeInBytes_Args;

/* fn[69] — dst_buffer is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Buffer* buffer;
    PJRT_Device* dst_device;
    PJRT_Buffer* dst_buffer;  /* out */
} PJRT_Buffer_CopyToDevice_Args;

/* fn[70] — event is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Buffer*              src;
    PJRT_Buffer_MemoryLayout* host_layout;   /* may be NULL for default */
    void*                     dst;
    size_t                    dst_size;
    PJRT_Event*               event;  /* out */
} PJRT_Buffer_ToHostBuffer_Args;

/* fn[95] — topology is OUT */
typedef struct {
    PJRT_HDR;
    PJRT_Client*              client;
    PJRT_TopologyDescription* topology;  /* out */
} PJRT_Client_TopologyDescription_Args;

/* fn[84] — platform_name / platform_name_size are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_TopologyDescription* topology;
    const char*               platform_name;       /* out */
    size_t                    platform_name_size;  /* out */
} PJRT_TopologyDescription_PlatformName_Args;

/* fn[85] — platform_version / platform_version_size are OUT */
typedef struct {
    PJRT_HDR;
    PJRT_TopologyDescription* topology;
    const char*               platform_version;       /* out */
    size_t                    platform_version_size;  /* out */
} PJRT_TopologyDescription_PlatformVersion_Args;

#endif /* TPU_PJRT_H */
