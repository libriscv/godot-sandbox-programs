# Native x86_64 Linux Toolchain
# This builds for the host architecture (x86_64) instead of RISC-V

# Use standard Linux system
set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR "x86_64")

# Not cross-compiling when building for host
set(CMAKE_CROSSCOMPILING FALSE)

# Use the native compilers from the system
find_program(CMAKE_C_COMPILER
    NAMES
        gcc
        cc
    PATHS
        ENV PATH
)

find_program(CMAKE_CXX_COMPILER
    NAMES
        g++
        c++
    PATHS
        ENV PATH
)

if (NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR
        "Native C/C++ compilers not found in PATH.\n"
        "Please install GCC/G++:\n"
        "  Linux: sudo dnf install gcc gcc-c++\n"
        "  macOS: brew install gcc\n"
    )
endif()

message(STATUS "Found native C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Found native C++ compiler: ${CMAKE_CXX_COMPILER}")

# Set architecture flags for x86_64
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64 -mtune=generic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64 -mtune=generic")

# Ensure we're building for the correct architecture
add_compile_options(-m64)

# Disable .d dependency files on Windows (not needed for Linux)
if (CMAKE_HOST_WIN32)
    set(CMAKE_C_LINKER_DEPFILE_SUPPORTED FALSE)
    set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED FALSE)
endif()
