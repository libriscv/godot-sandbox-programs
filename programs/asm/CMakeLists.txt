cmake_minimum_required(VERSION 3.10)
project(asmjit)

option(ZLIB_BUILD_EXAMPLES "" OFF)

include(FetchContent)
FetchContent_Declare(
  zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG develop
)
FetchContent_MakeAvailable(zlib)

# Add the RISC-V asmjit static library
add_library(riscv_asmjit STATIC IMPORTED)
if (ZIG_COMPILER)
	set_target_properties(riscv_asmjit PROPERTIES
		IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/zig-libRiscvAsmLib.a)
else()
	set_target_properties(riscv_asmjit PROPERTIES
		IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libRiscvAsmLib.a)
endif()
target_link_libraries(riscv_asmjit INTERFACE zlibstatic)

# Add the asmjit sandbox program
add_ci_program(asm
	main.cpp
)
target_link_libraries(asm PRIVATE riscv_asmjit)

if (ZIG_COMPILER)
	target_sources(asm PRIVATE extra/obstack.c)
endif()
