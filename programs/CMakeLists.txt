# Add the sandbox program CMake API
add_subdirectory(../ext/godot-sandbox/program/cpp/cmake cmake)

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
endfunction()

# Build all the examples
add_subdirectory(asm)
add_subdirectory(hello-world)
add_subdirectory(libtcc)
add_subdirectory(luajit)
add_subdirectory(mir)
add_subdirectory(robust_skin_weight_transfer)
