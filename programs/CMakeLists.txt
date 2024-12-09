# Fetch godot-sandbox repository (add_subdirectory is implicitly called)
include(FetchContent)
FetchContent_Declare(
	godot-sandbox
	GIT_REPOSITORY https://github.com/libriscv/godot-sandbox.git
	GIT_TAG        main
	GIT_PROGRESS   TRUE
	SOURCE_SUBDIR  "program/cpp/cmake"
)
FetchContent_MakeAvailable(godot-sandbox)

# Download the run-time generated API from the v0.1 release
file(DOWNLOAD
	https://github.com/libriscv/godot-sandbox-programs/releases/download/v0.1/generated_api.hpp
	${CMAKE_BINARY_DIR}/generated_api.hpp)
set_source_files_properties(${CMAKE_BINARY_DIR}/generated_api.hpp PROPERTIES GENERATED TRUE)

function (add_ci_program name)
	add_sandbox_program(${name} ${ARGN})
	set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
	target_include_directories(${name} PRIVATE ${CMAKE_BINARY_DIR})
	target_sources(${name} PRIVATE ${CMAKE_BINARY_DIR}/generated_api.hpp)
	if (STRIPPED)
		target_link_options(${name} PRIVATE -Wl,--strip-all)
	endif()
	if (USE_MOLD)
		if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
			message(FATAL_ERROR "MOLD requires GCC 12.0 or later")
		endif()
		target_link_options(${name} PRIVATE -fuse-ld=mold)
	endif()
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
