cmake_minimum_required(VERSION 3.10)
project(mirjit)

add_ci_program(mirjit
	main.cpp
)

set(LLVM_FOUND FALSE CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(libmir
	GIT_REPOSITORY https://github.com/vnmakarov/mir
	GIT_TAG        master
)
FetchContent_MakeAvailable(libmir)

target_link_libraries(mirjit PRIVATE mir_static)
target_include_directories(mirjit PRIVATE "${CMAKE_BINARY_DIR}/_deps/libmir-src")
