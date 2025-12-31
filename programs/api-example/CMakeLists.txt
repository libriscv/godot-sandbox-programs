cmake_minimum_required(VERSION 3.10)
project(api-example)

# Build an api-example RISC-V executable
add_ci_program(api_example
	main.cpp
)

