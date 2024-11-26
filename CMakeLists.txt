cmake_minimum_required(VERSION 3.10)
project (examples)

add_subdirectory(asm)
add_subdirectory(libtcc)
add_subdirectory(luajit)
add_subdirectory(mir)

set_target_properties(asm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(cjit PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(lua PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(mirjit PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
