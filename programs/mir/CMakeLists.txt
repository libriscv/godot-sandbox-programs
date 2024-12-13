cmake_minimum_required(VERSION 3.10)
project(mirjit)

set(LLVM_FOUND FALSE CACHE BOOL "" FORCE)
option(MIR_BUILD_TESTS "" OFF)
option(MIR_BUILD_UTILS "" OFF)

# Fetch the MIR library
include(FetchContent)
FetchContent_Declare(libmir
	GIT_REPOSITORY https://github.com/fwsGonzo/mir
	GIT_TAG        master
)
FetchContent_MakeAvailable(libmir)

# Build a RISC-V executable w/MIR
add_ci_program(mirjit
	main.cpp
)
target_link_libraries(mirjit PRIVATE mir_static)
target_include_directories(mirjit PRIVATE "${CMAKE_BINARY_DIR}/_deps/libmir-src")
