cmake_minimum_required(VERSION 3.10)
project(scenetree_passthrough)

add_ci_program(scenetree_passthrough
    main.cpp
    handlers.cpp
)

# Define EMBEDDED_RISCV flag for the RISC-V embedded environment (godot-sandbox)
# This enables headless execution without a display/TTY
target_compile_definitions(scenetree_passthrough PRIVATE EMBEDDED_RISCV)

# Alternative: Define STANDALONE_NATIVE for x86_64 native builds
# Do NOT define both - they are mutually exclusive
# Uncomment for standalone native (non-RISC-V, non-godot-sandbox):
# target_compile_definitions(scenetree_passthrough PRIVATE STANDALONE_NATIVE)
