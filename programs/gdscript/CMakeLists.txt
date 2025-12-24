cmake_minimum_required(VERSION 3.10)
project(gdscript_compiler)

add_subdirectory(${CMAKE_BINARY_DIR}/_deps/godot-sandbox-src/src/gdscript/compiler gdscript_compiler)

add_ci_program(gdscript
	main.cpp
)
target_link_libraries(gdscript PRIVATE gdscript_compiler)
