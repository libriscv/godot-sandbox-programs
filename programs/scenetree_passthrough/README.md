# JSON-RPC SceneTree Server for godot-sandbox

A complete SceneTree API passthrough server for RISC-V godot-sandbox that exposes all Godot SceneTree and Object methods with **exact API names** (no prefixes).

## Overview

This server provides direct RPC access to the Godot engine's SceneTree and Object APIs. All method names match the Godot API exactly - no prefixes, no abstraction.

**SceneTree methods** are called via `get_tree()` (SceneTree is a MainLoop, not a Node).

**Object/Node methods** are called via `node.call()`, `node.get()`, `node.set()`.

**Internal methods** starting with `_` (like `_process`, `_initialize`) are NOT exposed - they are Godot's internal virtual methods.

## Auto-Generated API Reference

See [API_REFERENCE.md](./API_REFERENCE.md) for the complete API documentation generated from the Godot 4.6.1 jsonl method definitions.

## Build

```bash
cd godot-sandbox-programs
./build.sh
```

## Usage Example

```gdscript
# Set up Sandbox in your Godot project:
# 1. Add a Node named "Sandbox" as child of root
# 2. Attach the jsonrpc_server binary or script

var sandbox = get_node("/root/Sandbox")

# Call API methods:
var result = sandbox.call("node_call", node_path, "method_name", args)
```

## C++ Examples

The Sandbox API can access Godot objects, properties, methods, and nodes directly from C++.

### Node paths

All node paths are relative to the node that has the sandbox program attached.

```cpp
#include <api.hpp>

int main() {
  // Current node (the node this program is attached to)
  Node current_node(".");

  // Equivalent helper for current node
  Node also_current = get_node();

  // Relative paths
  Node parent = get_node<Node>("..");
  Node sibling = get_node<Node>("../OtherNode");
  Node child = get_node<Node>("UI/Label");

  // Absolute path from SceneTree root
  Node absolute = get_node<Node>("/root/Main/Player");

  return 0;
}
```

### Current node access (`.` and `get_node()`)

```cpp
#include <api.hpp>

static Variant print_current_name() {
  Node current_a(".");
  Node current_b = get_node();

  print("Current A: ", current_a.get_name());
  print("Current B: ", current_b.get_name());
  return Variant();
}

int main() {
  ADD_API_FUNCTION(print_current_name, "void");
  halt();
}
```

## Architecture

```
LLM (Temporal Planner)
  -> JSON-RPC over ENet UDP
    -> godot-sandbox RPC Server
      -> SceneTree.get_tree().call(method, args)
      -> Node.call(path, method, args)
```

## Method Generation

The API reference is auto-generated from the Godot 4.6.1 JSONL definitions.

To regenerate the API documentation:

```bash
python3 generate_api.py
```

Then build the project:

```bash
cd godot-sandbox-programs && ./build.sh
```
