# ==================== Standalone SceneTree Passthrough ====================
# A standalone JSON-RPC 2.0 server that exposes Godot's SceneTree and Node API.
#
# This version can build:
#   1. standalone executable (TCP/stdio) - uses mock handlers
#   2. GDExtension shared library - uses GDExtension API to call Godot
#
# Build: cmake -S . -B ../../.build_standalone && make -C ../../.build_standalone
# Run: ./../../.build_standalone/bin/scenetree_passthrough_standalone --stdio
#
# Cross-platform targets:
#   - Linux (x86_64, aarch64)
#   - macOS (x86_64, arm64)  
#   - Windows (x86_64, arm64)
#
# RISC-V/ELF builds are handled by godot-sandbox (embedded mode).
# This standalone executable runs ON the target platform, not as RISC-V/ELF.

cmake_minimum_required(VERSION 3.15)
project(scenetree_passthrough_standalone LANGUAGES CXX)

# ==================== Project Structure ====================
get_filename_component(BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(COMMON_DIR "${BASE_DIR}/programs/scenetree_common" ABSOLUTE)

message(STATUS "Building Standalone SceneTree Passthrough")
message(STATUS "  Base dir: ${BASE_DIR}")
message(STATUS "  Common dir: ${COMMON_DIR}")

# ==================== Common JSON-RPC Library ====================
# Include the common sublibrary (JSON-RPC server framework + mock handlers)
# Mock handlers are used since the standalone executable doesn't have Godot API access.
# Set the handler implementation for scenetree_common (built by programs/CMakeLists.txt)
set(SCENETREE_HANDLERS "mock" CACHE STRING "Handler implementation to use")
# Note: The scenetree_common subdirectory is already built by programs/CMakeLists.txt.
# We just need the headers and the compiled library from the parent scope.
# Setting SCENETREE_HANDLERS makes the COMMON_DIR's CMakeLists use the mock handlers.
# include_directories(${COMMON_DIR}/include ${COMMON_DIR}/src)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==================== Compiler ====================
set(CMAKE_C_STANDARD 11)

# ==================== Sources ====================
set(STANDALONE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/standalone_main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/standalone_jsonrpc.cpp"
)

# ==================== Executable (Standalone TCP/stdio server) ====================
add_executable(scenetree_passthrough_standalone ${STANDALONE_SOURCES})
target_link_libraries(scenetree_passthrough_standalone PRIVATE scenetree_common)

# Include directories for standalone build
target_include_directories(scenetree_passthrough_standalone PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${BASE_DIR}/api"
    "${BASE_DIR}/ext"
)

# Compiler warnings
target_compile_options(scenetree_passthrough_standalone PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# ==================== Platform-Specific Settings ====================
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_compile_options(scenetree_passthrough_standalone PRIVATE
        -Wno-deprecated-declarations  # For macOS deprecated APIs
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    target_compile_options(scenetree_passthrough_standalone PRIVATE
        -Wno-unknown-pragmas
    )
endif()

# ==================== Output Directory ====================
set_target_properties(scenetree_passthrough_standalone PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ==================== GDExtension Shared Library (optional) ====================
# Uncomment to build a GDExtension/.dll/.so that can be loaded by Godot
# This provides the same API via GDExtension instead of TCP/stdio
#
# Usage:
#   add_library(scenetree_passthrough_gdextension SHARED gdextension_entry.cpp)
#   target_link_libraries(scenetree_passthrough_gdextension PRIVATE scenetree_common)
#
# Note: GDExtension mode embeds the JSON-RPC server directly in the Godot process,
# so it needs access to Godot's GDExtension API headers.

# Uncomment the following block to enable GDExtension build:
#
# add_library(scenetree_passthrough_gdextension SHARED
#     "${CMAKE_CURRENT_SOURCE_DIR}/gdextension_entry.cpp"
# )
# target_link_libraries(scenetree_passthrough_gdextension PRIVATE scenetree_common)
# target_include_directories(scenetree_passthrough_gdextension PRIVATE
#     "${CMAKE_CURRENT_SOURCE_DIR}"
#     "${BASE_DIR}/api"
#     "${BASE_DIR}/ext"
#     "${GODOT_CPP_INCLUDE_DIRS}"
# )
#
# if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
#     set_target_properties(scenetree_passthrough_gdextension PROPERTIES
#         SUFFIX ".dylib"
#         PREFIX "lib"
#     )
# elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
#     set_target_properties(scenetree_passthrough_gdextension PROPERTIES
#         SUFFIX ".dll"
#     )
# endif()
