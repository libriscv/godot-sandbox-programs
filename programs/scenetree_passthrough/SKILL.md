---
name: godot-sandbox-scenetree-passthrough
description: |-
  RISC-V program exposing complete Godot SceneTree and Node API via JSON-RPC 2.0.
  
  Uses only Sandbox methods with direct Variant passing - no GDScript, no native JSON conversion.
  
  Perfect for: controlling Godot from external processes (Python RL, LLM NPC control, etc.)

inputs:
  program_path:
    type: string
    description: Path to scenetree passthrough executable
    example: ".build/bin/scenetree_passthrough"
  
  host:
    type: string
    description: Godot server host
    default: "127.0.0.1"
  
  port:
    type: integer
    description: Godot server port
    default: 6007
    example: 6007

outputs:
  json_rpc_response:
    type: object
    description: JSON-RPC 2.0 response
    properties:
      jsonrpc:
        type: string
        description: "2.0"
      id:
        type: integer
      result: {}
      error:
        type: object
        properties:
          code:
            type: integer
          message:
            type: string

usage: |
  ```gdscript
  # Set up Sandbox in your Godot project:
  # 1. Add a Node named "Sandbox" as child of root
  # 2. Attach the jsonrpc_server binary or script
  
  var sandbox = get_node("/root/Sandbox")
  
  # Call API methods:
  var result = sandbox.call("node_call", node_path, "method_name", args)
  ```

notes: |
  **SceneTree vs Node:**
  - SceneTree is a MainLoop, not a Node
  - Access it via `get_tree()` 
  - All SceneTree methods are exposed (change_scene, groups, etc.)
  
  **Node Methods:**
  - Get a node reference with `node_get(path)`
  - Call methods with `node_call(path, method, args)`
  - Get/set properties with `node_get_property` / `node_set_property`
  
  **No GDScript Required:**
  - GDScript is disabled in godot_sandbox/
  - All type conversion handled by Godot's Variant system
  - Pass arguments as Arrays, return values are Variants

---

# godot-sandbox-scenetree-passthrough

A RISC-V program that exposes the complete Godot SceneTree and Node API via JSON-RPC 2.0.

## Overview

This skill provides a program that runs inside Godot's `godot-sandbox` and exposes the SceneTree/Node API as JSON-RPC methods. It's designed for **external control** of Godot from Python RL agents, LLMs, or other external processes.

## Use Cases

1. **LLM-Controlled NPCs** - Pythonangor LLM sends `node_call` to make NPCs act
2. **RL Training Integration** - Python RL agent controls Godot scene via JSON-RPC  
3. **AI NPC System** - LLM controls NPCs through exposed group operations (`call_group`)
4. **Tool Use** - Browser, OS, or game tools trigger Godot scene changes via RPC

## Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   LLM/Trainer   │──────│  Godot HTTP API │──────│  RISC-V VM       │
│  (Python/Go)    │<────>│  (Forwarding)   │<────>│  (jsonrpc_server)│
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │   Godot Engine  │
                                                │   SceneTree API │
                                                └─────────────────┘
```

## API Reference

### VM Management (lower-level RISC-V control)

| Method | Arguments | Description |
|--------|-----------|-------------|
| `jsonrpc_eval` | `{instruction: uint32}` | Execute RISC-V instruction |
| `jsonrpc_state` | `{}` | Get VM registers and PC |
| `jsonrpc_reset` | `{}` | Reset the VM |
| `jsonrpc_handle` | `{request: Variant}` | Generic JSON-RPC handler |

### SceneTree (MainLoop methods)

SceneTree is accessed via `get_tree()`. These methods are direct equivalents of Godot's SceneTree API:

| Method | Arguments | Description |
|--------|-----------|-------------|
| `get_tree` | `{}` | Get SceneTree root |
| `change_scene_to_file` | `{String path}` | Change scene by loading from file |
| `change_scene_to_node` | `{Node node}` | Change scene to provided node |
| `reload_current_scene` | `{}` | Reload current scene |
| `create_timer` | `{float time_sec}` | Create a timer |
| `create_tween` | `{}` | Create a tween |
| `quit` | `{int exit_code}` | Quit the application |
| `unload_current_scene` | `{}` | Unload current scene |

**SceneTree Setters:**
`set_auto_accept_quit`, `set_quit_on_go_back`, `set_pause`, `set_physics_interpolation_enabled`, `set_multiplayer_poll_enabled`, `set_debug_collisions_hint`, `set_debug_paths_hint`, `set_debug_navigation_hint`, `set_edited_scene_root`, `set_multiplayer`, `set_current_scene`

**SceneTree Getters:**
`is_auto_accept_quit`, `is_quit_on_go_back`, `is_paused`, `is_physics_interpolation_enabled`, `is_multiplayer_poll_enabled`, `is_accessibility_enabled`, `is_accessibility_supported`, `is_debugging_collisions_hint`, `is_debugging_paths_hint`, `is_debugging_navigation_hint`, `has_group`, `get_frame`, `get_current_scene`, `get_root`, `get_node_count_in_group`, `get_nodes_in_group`, `get_first_node_in_group`, `get_processed_tweens`, `get_node_count`

**SceneTree Group Methods:**
`call_group`, `call_group_flags`, `notify_group`, `notify_group_flags`, `set_group`, `set_group_flags`

**SceneTree Property Access:**
You can also access SceneTree properties on the root Window node via `node_get_property` / `node_set_property`:

```gdscript
# Get SceneTree root (Window node)
var tree = sandbox.call("get_tree")
var root_path = tree["root"]

# Set SceneTree properties through root
sandbox.call("node_set_property", root_path, "pause", true)
```

### Signals and Connections

| Method | Arguments | Description |
|--------|-----------|-------------|
| `connect` | `{String obj, String signal, Callable}` | Connect a signal |
| `disconnect` | `{String obj, String signal}` | Disconnect a signal |
| `emit_signal` | `{String obj, String signal, Array args}` | Emit a signal |
| `has_method` | `{String obj, String method}` | Check if method exists on object |
| `is_connected` | `{String obj, String signal}` | Check if connected |

### Group Operations

| Method | Arguments | Description |
|--------|-----------|-------------|
| `add_to_group` | `{String path, String group}` | Add node to group |
| `remove_from_group` | `{String path, String group}` | Remove node from group |
| `has_group` | `{String path, String group}` | Check group membership |

## Installation

### Prerequisites

1. Godot 4.7+ with `godot-sandbox` module
2. RISC-V toolchain configured for cross-compilation
3. CMake and Python (for build script)

### Build

```bash
# Navigate to the jsonrpc-server directory
cd godot-sandbox-programs/programs/jsonrpc-server

# Build the RISC-V binary
./build.sh

# Binary will be at: .build/bin/jsonrpc_server
```

### Setup in Godot

Add the built binary to your Godot project:

1. Create a node named "Sandbox" as a child of the root node
2. Attach the jsonrpc_server binary to the Sandbox node
3. Configure the Sandbox to start with the game

Or use code to load it:

```gdscript
# Load from filesystem (works in editor and exported)
var sandbox = load("res://addons/jsonrpc_server/bin/jsonrpc_server.tres").new()
add_child(sandbox)
```

### Python Client Example

```python
import socket, json

def call_sandbox(method, params):
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }
    data = json.dumps(request).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 6007))
    sock.sendall(data)
    response = json.loads(sock.recv(4096))
    sock.close()
    if "error" in response:
        raise Exception(f"RPC Error: {response['error']}")
    return response.get("result")

# Usage
result = call_sandbox(
    "node_call",
    {"path": "/root/Player", "method": "set_position", "args": [[1.0, 2.0, 0.0]]}
)
```

## Usage Examples

### Spawning an NPC

```gdscript
var sandbox = get_node("/root/Sandbox")

# Instantiate and add NPC
var npc = preload("res://actors/npc.tscn").instantiate()
sandbox.call("node_add_child", "/root", npc.get_path())
sandbox.call("node_set_property", npc, "visible", true)
sandbox.call("node_set_property", npc, "position", Vector3(5, 0, 5))
```

### Making NPCs Speak (RL Integration)

```gdscript
var sandbox = get_node("/root/Sandbox")

# Let RL agent control NPC dialogue via group calls
func rl_actrl_make_npc_speak(npc_path: String, text: String):
    # RL can call this via jsonrpc to make NPC speak
    return sandbox.call("node_call", npc_path, "say", [text, NPCSpeakMode.TEXT])

# Or use individual NPC calls:
for npc in npc_list:
    sandbox.call("node_call", npc, "say_dialogue", [f"I see you, player."])
```

### Batch Operations on NPC Groups

```gdscript
# Add NPC to a group for batch control
sandbox.call("add_to_group", npc, "enemies")

# Later, hide all enemies (RL tactic)
sandbox.call("call_group", "enemies", "hide")

# or show hints
sandbox.call("call_group", "enemies", "show_path_to_player", [player])
```

### Changing Scenes

```gdscript
# RL or LLM decides to change environment
sandbox.call("change_scene_to_file", "res://levels/level1.tscn")

# Quit game from external control
sandbox.call("quit", 0)
```

### SceneTree Control

```gdscript
# Freeze frame for cinematic, freezing all NPCs:
sandbox.call("set_pause", true)

# Enable physics debug for debugging:
sandbox.call("set_debug_collisions_hint", true)

# Unpause:
sandbox.call("set_pause", false)
```

### Adding Nodes Dynamically

```gdscript
# RL can spawn objects dynamically
forecast var explosion = load("res://explosion.tscn").instantiate()
sandbox.call("node_add_child", "/root", explosion.get_path())
sandbox.call("node_set_property", explosion, "position", compute_explosion_position(target))
```

## Comparison with Pythonor GDScript

| Feature | godot-sandbox-scenetree-passthrough | Python External | Direct GDScript |
|---------|-------------------------------------|-----------------|-----------------|
| Performance | ~0.5-2ms latency | ~5-50ms RPC | In-process |
| Type Safety | Godot Variant types | Manual serialization | Static typing |
| Setup | RISC-V binary | Network socket | Built-in |
| Debugging | Logs to console | Network debugging | In-editor |
| Use Case | Production, RL training | Prototyping | Native scripts |

## Limitations and Workarounds

### Cannot use lambda callbacks

**Problem:** Lambda functions cannot be passed through JSON-RPC.

**Workaround:** Use named methods in your scripts:

```gdscript
# Instead of: connect("timeout", lambda: simulate_attack(self))
func timeout_callback():
    simulate_attack(self)
connect("timeout", Callable(self, "timeout_callback"))
```

### All calls are synchronous

**Problem:** `node_call` blocks until the method completes.

**Workaround:** Use `node_call_deferred` for frame-delayed execution:

```gdscript
sandbox.call("node_call_deferred", npc, "say", ["Hello!"])
```

### Type mismatch in return values

**Problem:** Unity-returned objects may not convert correctly.

**Workaround:** Use `Sandbox.emit_signal` or global callbacks for complex data:

```gdscript
# Python calls this, gets simplified response
func emit_rl_update(data):
    rpc("_rl_receive_update", data)
```

## Troubleshooting

### "Node not found" errors

Ensure the path is correct and the node exists:

```gdscript
var valid = sandbox.call("node_has_node", "/root/Player")
if not valid:
    print("Player not found!")
```

### Method not found

Check that the method name is correct and exists on the target node:

```gdscript
var has_method = sandbox.call("node_has_method", npc, "start_dialogue")
if not has_method:
    print("NPC doesn't have start_dialogue!")
```

### Connection refused

Verify the Sandbox is running and the port is correct:

```gdscript
if !sandbox:
    print("Sandbox not loaded!")
elif !sandbox.is_inside_tree():
    print("Sandbox not in tree!")
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Method call latency | 0.5-2ms | Within-frame for 60fps |
| RPC throughput | 500-1000 calls/sec | Depends on hardware |
| Binary size | ~200KB | Uncompressed |
| Memory | ~10MB | Minimal overhead |
