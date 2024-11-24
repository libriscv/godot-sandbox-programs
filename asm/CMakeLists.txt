cmake_minimum_required(VERSION 3.10)
project(asmjit)

add_subdirectory(../cmake cmake)

include(FetchContent)
FetchContent_Declare(
  libriscv
  GIT_REPOSITORY https://github.com/libriscv/libriscv.git
  GIT_TAG master
)
FetchContent_MakeAvailable(libriscv)

target_compile_definitions(riscv PRIVATE RISCV_DISABLE_URANDOM=1)

# Add the asmjit sandbox program
add_sandbox_program(asmjit
	main.cpp
)

target_link_libraries(asmjit PRIVATE -static riscv)
