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

# Remove RISC-V flags for macOS builds (clang++ doesn't support RISC-V arch flags)
# These are added by godot-sandbox but we're building for host, not cross-compiling
if(APPLE)
	# Function to remove RISC-V flags from a target
	function(remove_riscv_flags target_name)
		if(TARGET ${target_name})
			get_target_property(OPTS ${target_name} COMPILE_OPTIONS)
			if(OPTS)
				list(FILTER OPTS EXCLUDE REGEX "^-march=.*")
				list(FILTER OPTS EXCLUDE REGEX "^-mabi=.*")
				set_target_properties(${target_name} PROPERTIES COMPILE_OPTIONS "${OPTS}")
			endif()
		endif()
	endfunction()
	
	# Remove flags from sandbox_api and any other targets
	remove_riscv_flags(sandbox_api)
	
	# Also override the variables to prevent them from being used in future targets
	set(RISCV_ARCH "" CACHE STRING "" FORCE)
	set(RISCV_ABI "" CACHE STRING "" FORCE)
endif()

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
	add_subdirectory(stablehlo-to-riscv)
endif()
