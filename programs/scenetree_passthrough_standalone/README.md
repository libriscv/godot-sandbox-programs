# SceneTree Passthrough (Standalone)

A standalone JSON-RPC 2.0 server that exposes Godot's SceneTree and Node API without requiring the full Godot C++ API or `godot-sandbox`.

## Overview

This package provides two build targets:

1. **Standalone Executable** (`scenetree_passthrough_standalone`) - A complete JSON-RPC server that can run independently and communicate with external clients over TCP or stdin/stdout.

2. **GDExtension Shared Library** (`scenetree_passthrough_gdextension.so`) - A Godot GDExtension module that integrates with Godot's scene system and provides SceneTree/Node API access via JSON-RPC.

```
┌─────────────────────────────────────────────────────────────┐
│                    External Client                            │
│  (Python, Node.js, C++, curl, etc.)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ TCP or stdin/stdout
                              ▼
                    ┌─────────────────┐
                    │   JSON-RPC      │
                    │   Server        │
                    │  (Standalone)   │──────► Godot Engine
                    └─────────────────┘
                              │
                              │ GDExtension (optional)
                              ▼
                    ┌─────────────────┐
                    │  Godot Scene    │
                    │  Tree / Nodes   │
                    └─────────────────┘
```

## Quick Start

### Build

```bash
cd godot-sandbox-programs/programs/scenetree_passthrough_standalone

# Native x86_64
cmake -S . -B ../../.build && make -C ../../.build

# With RISC-V toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=../../cmake/riscv64-standalone.cmake -S . -B ../../.build_riscv
make -C ../../.build_riscv
```

### Run (Standalone Executable)

**TCP Mode (default):**
```bash
# Start server on port 7777
./godot-sandbox-programs/.build_standalone/bin/scenetree_passthrough_standalone --port 7777

# Connect with curl
curl -d '{"jsonrpc":"2.0","method":"list_methods","id":1}' http://localhost:7777
```

**Stdio Mode:**
```bash
# Start server with stdio mode (for pipes)
./scenetree_passthrough_standalone --stdio

# Pipe JSON-RPC requests
echo '{"jsonrpc":"2.0","method":"list_methods","id":1}' | ./scenetree_passthrough_standalone --stdio
```

### GDExtension Integration

```bash
# Build the GDExtension library
cmake -S . -B ../../.build_gdextension && make -C ../../.build_gdextension

# Copy to Godot project
cp godot-sandbox-programs/.build_gdextension/libscenetree_passthrough.so my_godot_project/addons/
```

## JSON-RPC API

### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "method_name",
  "params": ["argument1", 42, true],
  "id": 1
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "result": "result_value",
  "id": 1
}
```

### Error Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": "method_name"
  },
  "id": 1
}
```

### Available Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `list_methods` | none | Array | List all available methods |
| `get_tree` | none | Object | Get scene tree status |
| `get_root` | none | Node | Get root node |
| `has_group` | String `group_name` | Bool | Check if group exists |
| `get_nodes_in_group` | String `group_name` | Array | Get nodes in group |
| `set_pause` | Bool `paused` | String | Set pause state |
| `is_paused` | none | Bool | Get pause state |
| `get_current_scene` | none | Object | Get current scene |
| `quit` | none | String | Quit the application |

## Command Line Options

```
Usage: ./scenetree_passthrough_standalone [options]

Standalone SceneTree Passthrough JSON-RPC Server

Options:
  -p, --port <port>    TCP port to listen on (default: 7777)
  -s, --stdio          Use stdin/stdout mode (for pipes/fifos)
  -l, --log <file>     Log to file
  -h, --help           Show this help
```

## Architecture

### Directory Structure

```
programs/scenetree_passthrough_standalone/
├── CMakeLists.txt              # Build configuration
├── standalone_jsonrpc.cpp       # Main executable implementation
├── gdextension_entry.cpp           # GDExtension shared library entry point
├── common/
│   ├── jsonrpc_server.hpp      # JSON-RPC server framework (shared)
│   ├── jsonrpc_server.cpp      # JSON-RPC server implementation (shared)
│   └── handler_base.hpp        # Base class for handlers (shared)
└── api/
    └── *.jsonl                  # Godot API definitions (for handler generation)
```

### Key Components

#### 1. `jsonrpc_server.{hpp,cpp}`

Provides the core JSON-RPC 2.0 server infrastructure:

- **TCP Server Mode**: Listens on a socket, accepts multiple concurrent connections via threads
- **Stdio Mode**: Reads from stdin, writes to stdout (for Unix pipes and subprocess communication)
- **Request Parsing**: Parses JSON-RPC 2.0 requests, extracts method name and parameters
- **Response Formatting**: Formats JSON-RPC 2.0 responses with proper error handling
- **Logging**: Optional file logging for debugging

#### 2. `handler_base.hpp`

Provides utility classes for building JSON responses:

- `HandlerResult` struct for returning results or errors
- JSON building helpers (`json_add_string`, `json_add_int`, `json_add_bool`, etc.)
- String escaping utilities

#### 3. `standalone_jsonrpc.cpp`

The main executable that implements SceneTree/Node method handlers:

- Uses the shared `jsonrpc_server` infrastructure
- Implements handlers for SceneTree and Node methods
- Can run independently as a TCP/stdio server

#### 4. `gdextension_entry.cpp`

The GDExtension/GDExtension entry point that:

- Registers with Godot's class system
- Exposes SceneTree/Node methods via GDExtension
- Bridges to the shared JSON-RPC infrastructure

## Handler Implementation Pattern

All handlers follow the same pattern:

```cpp
static const char* handle_method_name(const char* params_json, int id) {
    // 1. Extract parameters from params_json
    const char* param = json_extract_string(params_json, "param_name");
    
    // 2. Validate parameters (return NULL for error)
    if (!param) {
        return NULL; // Will trigger "Invalid params" error
    }
    
    // 3. Call Godot API (or mock for standalone demo)
    const char* result = call_godot_method(param);
    
    // 4. Return JSON string result
    return result;
}
```

## Cross-Compilation

### RISC-V 64-bit

```bash
# Fedora
sudo dnf install gcc-riscv64-linux-gnu gcc-c++-riscv64-linux-gnu

# Ubuntu  
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

# Build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv64-standalone.cmake -S . -B ../.build_riscv
make -C ../.build_riscv
```

### ARM64 (aarch64)

```bash
# Install cross-compiler
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-standalone.cmake -S . -B ../.build_arm64
make -C ../.build_arm64
```

## Testing

### Unit Tests

```bash
# Run the executable test
./godot-sandbox-programs/.build_standalone/bin/scenetree_passthrough_standalone --stdio <<< '{"jsonrpc":"2.0","method":"list_methods","id":1}'

# Expected output:
# {"jsonrpc":"2.0","result":{"methods":["get_root","has_group",...]},"id":1}
```

### TCP Test

```bash
# Start server in background
./scenetree_passthrough_standalone --port 17777 &
SERVER_PID=$!

# Send test request
curl -s -d '{"jsonrpc":"2.0","method":"list_methods","id":1}' http://localhost:17777

# Kill server
kill $SERVER_PID
```

## Godot Integration Example

```gdscript
extends Node

# Load the native library
var NativeLib = preload("res://addons/scenetree_passthrough/scenetree_passthrough.gdns")
var rpc_server = NativeLib.new()

func _ready():
    # Start the RPC server on port 7777
    rpc_server.start(7777)
    
    # Register as an RPC client to send requests
    # (You would implement an RPC client class)
    rpc_client.connect_to_server("127.0.0.1", 7777)

func pause_game():
    # Call set_pause via JSON-RPC
    var result = rpc_client.call_method("set_pause", [true])
    print("Pause result: ", result)

func check_paused():
    # Call is_paused via JSON-RPC
    var result = rpc_client.call_method("is_paused", [])
    print("Is paused: ", result)
```

## Extending with New Methods

### For Standalone Executable

1. Add handler function:

```cpp
static const char* handle_my_custom_method(const char* params_json, int id) {
    // Parse params
    // Call logic
    // Return JSON result
}
```

2. Add to dispatch table:

```cpp
static const struct {
    const char* method;
    const char* (*handler)(const char* params_json, int id);
} handler_table[] = {
    // ... existing entries ...
    {"my_custom_method", handle_my_custom_method},
    {NULL, NULL}
};
```

### For GDExtension Integration

1. Add method to `method_info` array in `gdextension_entry.cpp`
2. Implement the handler (same as standalone)
3. Rebuild the GDExtension library

## Build Matrix

| Platform | Executable | GDExtension | Status |
|----------|-----------|----------|--------|
| Linux x86_64 | Yes | Yes | Build & Tested |
| Linux RISC-V | Yes | Yes | Build & Untested |
| Linux ARM64 | Yes | Yes | Build & Untested |
| Windows x64 | No | Yes | Not yet ported |
| macOS x64 | No | Yes | Not yet ported |

## Performance Notes

- The standalone server uses threads for concurrent connections
- JSON parsing is done manually for minimal overhead
- Handler dispatch uses linear search (O(n) for n methods)
- For production, consider a hash table for O(1) dispatch

## License

MIT License - see LICENSE file for details.

## See Also

- [godot-sandbox-programs](https://github.com/V-Sekai/godot-sandbox-programs) - The parent project containing the embedded version
- [GDExtension API](https://docs.godotengine.org/en/stable/tutorials/bindings/gdextension/gdextension_c.html) - Godot's C API for extensions
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification) - JSON-RPC standard
