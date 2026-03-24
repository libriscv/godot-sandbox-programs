# JSON-RPC Common Sublibrary

This directory contains shared infrastructure for the SceneTree Passthrough system.

## Purpose

The `common/` sublibrary provides reusable JSON-RPC 2.0 infrastructure that powers both:

1. **`scenetree_passthrough`** - Embedded mode (godot-sandbox)
   - Runs inside the godot-sandbox RISC-V environment
   - Exposes Godot's SceneTree/Node API via JSON-RPC over TCP
   - Uses Godot C++ API for actual SceneTree operations

2. **`scenetree_passthrough_standalone`** - Standalone mode
   - Runs as a standalone executable (or GDNative module)
   - Exposes SceneTree/Node API via JSON-RPC over TCP or stdio
   - Designed to work in RISC-V environments without full Godot scene

## Files

| File | Description |
|------|-------------|
| `jsonrpc_server.hpp` | C headers for JSON-RPC 2.0 server framework |
| `jsonrpc_server.cpp` | Implementation of TCP/stdio server loop and request parsing |
| `handler_base.hpp` | Base handler infrastructure and JSON building utilities |

## Building

The common sublibrary is automatically included when building either frontend:

```bash
# From godot-sandbox-programs/
cmake -S . -B .build && make -C .build
```

## Integration

Both frontends link against this sublibrary:

```cmake
add_subdirectory(../../common ${CMAKE_BINARY_DIR}/common_jsonrpc)
add_library(jsonrpc_common STATIC jsonrpc_server.cpp)
target_link_libraries(my_target PRIVATE jsonrpc_common)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    JSON-RPC Common Layer                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  jsonrpc_server.hpp/cpp  - Core server loop, TCP/stdio  │ │
│  │  handler_base.hpp       - JSON building, error handling │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  Embedded    │   │  Standalone  │   │   GDNative   │
   │  (godot-      │   │  Executable  │   │   (.so)      │
   │  sandbox)     │   │              │   │              │
   └──────────────┘   └──────────────┘   └──────────────┘
```

## License

This code is part of the SceneTree Passthrough system, licensed for use with godot-sandbox.
