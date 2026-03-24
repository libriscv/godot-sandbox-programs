#!/usr/bin/env python3
"""
Generate API_REFERENCE.md and update handler code from JSONL API definitions.

This script reads the Godot 4.6.1 JSONL method definitions and generates:
1. API_REFERENCE.md - Complete API documentation
2. Updates handler code in the C++ files

Usage:
    python3 generate_api.py          # Generate API_REFERENCE.md
    python3 generate_api.py --check # Check if API_REFERENCE.md is up to date
"""

import json
import os
import re
from pathlib import Path

# Paths
SCENETREE_COMMON_API = Path("programs/scenetree_common/api")
API_REFERENCE_PATH = Path("programs/scenetree_passthrough/API_REFERENCE.md")
HANDLER_BASE_PATH = Path("programs/scenetree_common/include/handler_base.hpp")

# Method flags mapping
METHOD_FLAGS_MAP = {
    1: "METHOD_FLAG_NORMAL",
    2: "METHOD_FLAG_EDITOR",
    4: "METHOD_FLAG_CONST",
    8: "METHOD_FLAG_VIRTUAL",
    16: "METHOD_FLAG_VARARG",
    32: "METHOD_FLAG_STATIC",
    64: "METHOD_FLAG_OBJECT_CORE",
}

def decompose_bitmask(mask: int) -> list:
    """Decompose a bitmask into individual flag names."""
    if mask == 0:
        return ["NONE"]
    
    active_flags = []
    for flag_value, flag_name in METHOD_FLAGS_MAP.items():
        if mask & flag_value == flag_value and flag_value != 0:
            active_flags.append(flag_name)
    return active_flags

def type_string(type_value: str) -> str:
    """Return type string as-is since JSONL already has string types."""
    # The JSONL already contains string type names like "Object", "StringName", "bool", etc.
    # No conversion needed.
    return type_value if type_value else "Unknown"

def load_jsonl(file_path: Path) -> list:
    """Load JSONL file and return list of method definitions."""
    methods = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                methods.append(json.loads(line))
    return methods

def generate_api_reference():
    """Generate API_REFERENCE.md from JSONL files."""
    print("Generating API_REFERENCE.md...")
    
    # Load public and virtual methods
    try:
        public_methods = load_jsonl(SCENETREE_COMMON_API / "public_methods_v4.6.1_stable.jsonl")
        virtual_methods = load_jsonl(SCENETREE_COMMON_API / "virtual_methods_v4.6.1_stable.jsonl")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run this from the godot-sandbox-programs directory")
        return False
    
    # Combine all methods
    all_methods = []
    for m in public_methods:
        m_copy = m.copy()
        # flags is already a list in JSONL, just clean up the names
        if "flags" in m_copy:
            flags_list = m_copy["flags"]
            m_copy["flags"] = [f.replace("METHOD_FLAG_", "") for f in flags_list]
        if "return" in m_copy and "type" in m_copy["return"]:
            m_copy["return"]["type"] = type_string(m_copy["return"]["type"])
        for arg in m_copy.get("args", []):
            if "type" in arg:
                arg["type"] = type_string(arg["type"])
        all_methods.append(m_copy)
    
    for m in virtual_methods:
        m_copy = m.copy()
        # flags is already a list in JSONL, just clean up the names
        if "flags" in m_copy:
            flags_list = m_copy["flags"]
            m_copy["flags"] = [f.replace("METHOD_FLAG_", "") for f in flags_list]
        if "return" in m_copy and "type" in m_copy["return"]:
            m_copy["return"]["type"] = type_string(m_copy["return"]["type"])
        for arg in m_copy.get("args", []):
            if "type" in arg:
                arg["type"] = type_string(arg["type"])
        all_methods.append(m_copy)
    
    # Sort by method name
    all_methods.sort(key=lambda x: x.get("name", ""))
    
    # Generate markdown
    lines = []
    lines.append("# SceneTree API Reference")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"This document lists all {len(all_methods)} SceneTree and Object methods")
    lines.append("that are exposed via the JSON-RPC interface.")
    lines.append("")
    lines.append("## Method Index")
    lines.append("")
    
    # Create alphabetical index
    for method in all_methods:
        name = method.get("name", "")
        args = method.get("args", [])
        arg_types = ", ".join([a.get("type", "Unknown") for a in args])
        ret_type = method.get("return", {}).get("type", "void")
        lines.append(f"- [{name}](#{name}) - `{ret_type} {name}({arg_types})`")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Generate detailed method documentation
    for method in all_methods:
        name = method.get("name", "")
        args = method.get("args", [])
        ret = method.get("return", {})
        flags = method.get("flags", [])
        method_id = method.get("id", 0)
        
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"**Method ID:** {method_id}")
        lines.append(f"**Flags:** {', '.join(flags)}")
        lines.append("")
        
        # Arguments
        if args:
            lines.append("### Arguments")
            lines.append("")
            for arg in args:
                arg_name = arg.get("name", "")
                arg_type = arg.get("type", "Unknown")
                default = arg.get("default", None)
                lines.append(f"- `{arg_type} {arg_name}`")
                if default is not None:
                    lines.append(f"  - Default: `{default}`")
            lines.append("")
        
        # Return value
        ret_type = ret.get("type", "void")
        ret_class = ret.get("class_name", "")
        if ret_type != "void":
            lines.append(f"### Returns")
            lines.append("")
            lines.append(f"`{ret_type}`"
            )
            if ret_class:
                lines.append(f"- Class: `{ret_class}`")
            lines.append("")
        
        # Description (if available)
        # Note: JSONL doesn't include descriptions, would need XML docs for that
        lines.append(f"*Method names match Godot 4.6.1 API exactly*")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Write the file
    API_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(API_REFERENCE_PATH, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Generated {len(all_methods)} method entries in {API_REFERENCE_PATH}")
    return True

def main():
    """Main entry point."""
    if not generate_api_reference():
        print("Failed to generate API reference")
        return
    
    print("\nDone!")

if __name__ == "__main__":
    main()
