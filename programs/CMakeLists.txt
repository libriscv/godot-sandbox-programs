# Fetch godot-sandbox repository (add_subdirectory is implicitly called)
include(FetchContent)
FetchContent_Declare(
	godot-sandbox
	GIT_REPOSITORY https://github.com/libriscv/godot-sandbox.git
	GIT_TAG        main
	GIT_PROGRESS   TRUE
	GIT_SHALLOW    TRUE
	GIT_SUBMODULES ""
	SOURCE_SUBDIR  "program/cpp/cmake"
)
FetchContent_MakeAvailable(godot-sandbox)

function (add_ci_program name)
	add_sandbox_program(${name} ${ARGN})
	set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endfunction()

# Build the trivial examples
add_subdirectory(hello-world)


# Build the examples that require full Linux/C++ support
if (NOT EMBEDDED_RISCV)
	add_subdirectory(asm)
	add_subdirectory(gdscript)
	add_subdirectory(libtcc)
	add_subdirectory(luajit)
	add_subdirectory(mir)
	add_subdirectory(robust_skin_weight_transfer)
	# NOTE: scenetree_passthrough (embedded) was disabled in PR #17.
	# The handlers.cpp was replaced with mocks/handlers.cpp.
	# Use scenetree_passthrough_standalone instead.
	# add_subdirectory(scenetree_passthrough)
endif()

# ==================== Common JSON-RPC Sublibrary ====================
# Shared infrastructure for both embedded and standalone modes.
# Path: programs/scenetree_common (inside programs/)
#
# Set handler implementation for the common sublibrary:
#   - "mock" for standalone/stdio (no Godot dependency)
#   - "real" for embedded godot-sandbox (uses Godot C++ API)  
#   - "none" if frontend provides its own handlers
set(SCENETREE_HANDLERS "mock" CACHE STRING "Handler implementation: mock, real, or none")

add_subdirectory(${CMAKE_SOURCE_DIR}/programs/scenetree_common ${CMAKE_BINARY_DIR}/common_jsonrpc)

# ==================== Standalone SceneTree Passthrough ====================
# Builds a standalone executable (TCP/stdio server) using the common JSON-RPC sublibrary.
# This variant builds even with EMBEDDED_RISCV enabled, as it's designed to run
# in the RISC-V environment without requiring a full Godot scene.
# Note: The standalone executable runs ON the target platform (Linux/macOS/Windows)
# and is NOT a RISC-V/ELF binary. RISC-V/ELF builds are handled by godot-sandbox.
if (NOT EMBEDDED_RISCV OR TRUE)  # Always build standalone for now
	add_subdirectory(scenetree_passthrough_standalone)
endif()
