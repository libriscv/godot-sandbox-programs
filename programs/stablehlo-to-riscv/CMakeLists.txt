cmake_minimum_required(VERSION 3.10)
project(stablehlo-to-riscv)

# Build a stablehlo-to-riscv RISC-V executable
add_ci_program(stablehlo_to_riscv
	main.cpp
	stablehlo_parser.cpp
	cpp_generator.cpp
	riscv_compiler.cpp
)

