# CompileOptionsProto wire format (task 1.1 findings)

Decoded from `compile_opts.pb` / `compile_opts_n4.pb` (JAX-generated, 492 bytes each).

```
CompileOptionsProto
└─ field 3 (len, 489B): ExecutableBuildOptionsProto
   ├─ field 1  (varint): device_ordinal = -1   (encoded as 2^64-1, 10 bytes)
   ├─ field 3  (len, 461B): DebugOptionsProto  — default XLA flags; identical
   │                        across both blobs (treated as opaque)
   ├─ field 4  (varint): num_replicas          ← ONLY difference (1 vs 4)
   ├─ field 5  (varint): num_partitions = 1
   ├─ field 6  (varint): use_spmd_partitioning (absent in fixtures; bool)
   ├─ field 12 (len, 1B = 0x00): submessage, defaults
   ├─ field 18 (len, 1B = 0x00): submessage, defaults
   └─ field 23 (varint): 1
```

Notes:
- Negative device_ordinal uses standard non-zigzag int64 varint (10 bytes of 0xff + 0x01).
- Byte-identical re-encoding requires emitting fields in the order above and
  embedding the 461-byte debug_options verbatim (stored as a static array in
  `cpp/compile_opts.hpp`).
- Minimal encoding (no debug_options / 12 / 18 / 23) is also accepted by
  libtpu's PJRT compile path — verified by the oracle test (task 1.3).
