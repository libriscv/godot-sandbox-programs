cmake_minimum_required(VERSION 3.13.4)
project(test_mlir)

# Try to find Homebrew's LLVM/MLIR installation
set(HOMEBREW_PREFIX "/opt/homebrew")
if(EXISTS "${HOMEBREW_PREFIX}")
    set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" ${CMAKE_PREFIX_PATH})
endif()

find_package(MLIR QUIET)
find_package(LLVM QUIET)

if(MLIR_FOUND AND LLVM_FOUND)
    message(STATUS "Found MLIR/LLVM in Homebrew")

    add_executable(test_mlir ../test_mlir.cpp)

    target_link_libraries(test_mlir PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        LLVMCore
        LLVMSupport
    )

    target_compile_definitions(test_mlir PRIVATE USE_MLIR=1)
else()
    message(STATUS "MLIR/LLVM not found, building minimal test")

    add_executable(test_mlir ../test_mlir.cpp)
    target_compile_definitions(test_mlir PRIVATE USE_MLIR=0)
endif()