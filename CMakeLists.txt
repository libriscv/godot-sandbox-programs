cmake_minimum_required(VERSION 3.14)
project (godot-sandbox-programs)

option(USE_MOLD  "Build with MOLD" OFF)
option(STRIPPED  "Strip executables" OFF)

if (${CMAKE_C_COMPILER} MATCHES "riscv64-unknown-elf*")
	set(EMBEDDED_RISCV 1)
	message(STATUS "Building for embedded RISC-V")
endif()

add_subdirectory(ext)
add_subdirectory(programs)
