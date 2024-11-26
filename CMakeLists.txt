cmake_minimum_required(VERSION 3.10)
project (examples)

# Add the sandbox program CMake API
add_subdirectory(cmake)
function (add_ci_program name)
	add_sandbox_program(${name} ${ARGN})
	set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endfunction()

# Build all the examples
add_subdirectory(asm)
add_subdirectory(libtcc)
add_subdirectory(luajit)
add_subdirectory(mir)
