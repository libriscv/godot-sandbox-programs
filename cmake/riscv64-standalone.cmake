# ==================== RISC-V 64-bit Toolchain for Standalone Build ====================
# Cross-compile the scenetree_passthrough_standalone for RISC-V 64-bit
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv64-standalone.cmake -S . -B ../.build_riscv
#   make -C ../.build_riscv
#
# Prerequisites:
#   - riscv64-unknown-elf-gcc or riscv64-linux-gnu-gcc installed
#   - For Fedora: sudo dnf install gcc-riscv64-linux-gnu gcc-c++-riscv64-linux-gnu
#   - For Ubuntu: sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Toolchain prefixes
set(RISCV_PREFIX "riscv64-linux-gnu")  # Use linux-gnu for glibc, not unknown-elf

# Find compiler
find_program(RISCV_C_COMPILER ${RISCV_PREFIX}-gcc)
find_program(RISCV_CXX_COMPILER ${RISCV_PREFIX}-g++)

if(NOT RISCV_C_COMPILER OR NOT RISCV_CXX_COMPILER)
    message(FATAL_ERROR 
        "RISC-V toolchain not found. Please install ${RISCV_PREFIX}-gcc and ${RISCV_PREFIX}-g++\n"
        "  Fedora: sudo dnf install gcc-riscv64-linux-gnu gcc-c++-riscv64-linux-gnu\n"
        "  Ubuntu: sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu"
    )
endif()

# Set compilers
set(CMAKE_C_COMPILER ${RISCV_C_COMPILER})
set(CMAKE_CXX_COMPILER ${RISCV_CXX_COMPILER})

# Sysroot - use system paths by default
set(CMAKE_SYSROOT /)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Target environment variables
set(ENV{PKG_CONFIG} "${RISCV_PREFIX}-pkg-config")
set(ENV{PKG_CONFIG_PATH} "${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig")

# Compiler flags for RISC-V
set(RISCV_C_FLAGS "-march=rv64imafdc_zicsr_zifencei -mabi=lp64d")
set(RISCV_CXX_FLAGS "-march=rv64imafdc_zicsr_zifencei -mabi=lp64d")

# Linker flags
set(RISCV_LINK_FLAGS "-Wl,--gc-sections -Wl,--as-needed")

# Apply flags to all targets
add_compile_options(${RISCV_C_FLAGS} ${RISCV_CXX_FLAGS})
add_link_options(${RISCV_LINK_FLAGS})

# Use absolute paths for find_* commands
set(CMAKE_FIND_USE_CMAKE_PATH TRUE)
set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH TRUE)

# Find pthread on RISC-V
find_package(Threads REQUIRED)

message(STATUS "RISC-V Toolchain Configured")
message(STATUS "  Compiler: ${RISCV_C_COMPILER}")
message(STATUS "  C Flags: ${RISCV_C_FLAGS}")
message(STATUS "  CXX Flags: ${RISCV_CXX_FLAGS}")
