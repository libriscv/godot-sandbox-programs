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
	add_subdirectory(libtcc)
	add_subdirectory(luajit)
	add_subdirectory(mir)
	add_subdirectory(robust_skin_weight_transfer)
endif()
