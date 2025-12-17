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

# Patch native.cpp to make string functions weak to allow duplicates in cross-compilation (no longer needed with -ffreestanding)
# file(READ "${CMAKE_BINARY_DIR}/_deps/godot-sandbox-src/program/cpp/docker/api/native.cpp" native_content)
# # Add .weak to CREATE_SYSCALL and CREATE_SYSCALL_STRCMP
# string(REPLACE "#define CREATE_SYSCALL(name, syscall_id)                 \
# 	__asm__(\".pushsection .text\\n\"                  \
# 			\".global \" #name \"\\n\"            \
# 			\".type \" #name \", @function\\n\"   \
# 			\"\" #name \":\\n\"                   \
# 			\"	li a7, \" STR(syscall_id) \"\\n\"       \
# 			\"	ecall\\n\" \
# 			\"	ret\\n\"   \
# 			\".popsection .text\\n\")"
# 			 "#define CREATE_SYSCALL(name, syscall_id)                 \
# 	__asm__(\".pushsection .text\\n\"                  \
# 			\".weak \" #name \"\\n\"              \
# 			\".global \" #name \"\\n\"            \
# 			\".type \" #name \", @function\\n\"   \
# 			\"\" #name \":\\n\"                   \
# 			\"	li a7, \" STR(syscall_id) \"\\n\"       \
# 			\"	ecall\\n\" \
# 			\"	ret\\n\"   \
# 			\".popsection .text\\n\")"
# 			 native_content ${native_content})
# string(REPLACE "#define CREATE_SYSCALL_STRCMP(name, syscall_id) \
# 	__asm__(\".pushsection .text\\n\"              \
# 			\".global \" #name \"\\n\"            \
# 			\".type \" #name \", @function\\n\"   \
# 			\"\" #name \":\\n\"                   \
# 			\"   li a2, 4096\\n\"               \
# 			\"	li a7, \" STR(syscall_id) \"\\n\"       \
# 			\"	ecall\\n\" \
# 			\"	ret\\n\"   \
# 			\".popsection .text\\n\")"
# 			 "#define CREATE_SYSCALL_STRCMP(name, syscall_id) \
# 	__asm__(\".pushsection .text\\n\"              \
# 			\".weak \" #name \"\\n\"              \
# 			\".global \" #name \"\\n\"            \
# 			\".type \" #name \", @function\\n\"   \
# 			\"\" #name \":\\n\"                   \
# 			\"   li a2, 4096\\n\"               \
# 			\"	li a7, \" STR(syscall_id) \"\\n\"       \
# 			\"	ecall\\n\" \
# 			\"	ret\\n\"   \
# 			\".popsection .text\\n\")"
# 			 native_content ${native_content})
# file(WRITE "${CMAKE_BINARY_DIR}/_deps/godot-sandbox-src/program/cpp/docker/api/native.cpp" ${native_content})

function (add_ci_program name)
	add_sandbox_program(${name} ${ARGN})
	set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
	if (CMAKE_CROSSCOMPILING)
		target_link_options(${name} PRIVATE "-nodefaultlibs" "-lc++")
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
	add_subdirectory(stablehlo-to-riscv)
endif()
