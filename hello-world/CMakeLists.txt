cmake_minimum_required(VERSION 3.10)
project(hello-world)

# Build a hello-world RISC-V executable
add_ci_program(hello_world
	main.cpp
)
