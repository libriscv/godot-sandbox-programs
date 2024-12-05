cmake_minimum_required(VERSION 3.14)
project (godot-sandbox-programs)

option(USE_MOLD  "Build with MOLD" OFF)
option(STRIPPED  "Strip executables" OFF)

add_subdirectory(ext)
add_subdirectory(programs)
