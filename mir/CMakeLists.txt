cmake_minimum_required(VERSION 3.10)
project(cjit)

add_subdirectory(../cmake cmake)

add_sandbox_program(cjit
	main.cpp
)

set(LLVM_FOUND FALSE CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(libmir
	GIT_REPOSITORY https://github.com/vnmakarov/mir
	GIT_TAG        master
)
FetchContent_MakeAvailable(libmir)

target_link_libraries(cjit PRIVATE mir_static)
target_include_directories(cjit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/_deps/libmir-src")
