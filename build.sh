#!/bin/bash
# RISC-V Build Script
# Sets up RISC-V toolchain and builds the project

set -e

# Create symlinks for riscv64-linux-gnu tools
# These are needed for proper toolchain identification
create_riscv_symlinks() {
    if [ ! -d "riscv64-toolchain/bin" ]; then
        echo "RISC-V toolchain not found. Please run with toolchain in PATH or download it."
        exit 1
    fi
    
    pushd riscv64-toolchain/bin
    
    # Remove existing symlinks first to avoid errors
    for tool in riscv64-linux-gnu-gcc riscv64-linux-gnu-g++ riscv64-linux-gnu-ld \
                 riscv64-linux-gnu-ar riscv64-linux-gnu-ranlib riscv64-linux-gnu-strip \
                 riscv64-linux-gnu-as riscv64-linux-gnu-nm riscv64-linux-gnu-objdump \
                 riscv64-linux-gnu-objcopy riscv64-linux-gnu-readelf riscv64-linux-gnu-addr2line \
                 riscv64-linux-gnu-elfedit riscv64-linux-gnu-c++ riscv64-linux-gnu-cpp \
                 riscv64-linux-gnu-gcov riscv64-linux-gnu-gcov-dump riscv64-linux-gnu-gcov-tool \
                 riscv64-linux-gnu-cc riscv64-linux-gnu-c++; do
        if [ -f "$tool" ] || [ -L "$tool" ]; then
            rm -f "$tool"
        fi
    done
    
    # Create standard riscv64-linux-gnu symlinks to the bootlin toolchain
    ln -sf riscv64-buildroot-linux-gnu-gcc riscv64-linux-gnu-gcc
    ln -sf riscv64-buildroot-linux-gnu-g++ riscv64-linux-gnu-g++
    ln -sf riscv64-buildroot-linux-gnu-ld riscv64-linux-gnu-ld
    ln -sf riscv64-buildroot-linux-gnu-ar riscv64-linux-gnu-ar
    ln -sf riscv64-buildroot-linux-gnu-ranlib riscv64-linux-gnu-ranlib
    ln -sf riscv64-buildroot-linux-gnu-strip riscv64-linux-gnu-strip
    ln -sf riscv64-buildroot-linux-gnu-as riscv64-linux-gnu-as
    ln -sf riscv64-buildroot-linux-gnu-nm riscv64-linux-gnu-nm
    ln -sf riscv64-buildroot-linux-gnu-objdump riscv64-linux-gnu-objdump
    ln -sf riscv64-buildroot-linux-gnu-objcopy riscv64-linux-gnu-objcopy
    ln -sf riscv64-buildroot-linux-gnu-readelf riscv64-linux-gnu-readelf
    ln -sf riscv64-buildroot-linux-gnu-addr2line riscv64-linux-gnu-addr2line
    ln -sf riscv64-buildroot-linux-gnu-elfedit riscv64-linux-gnu-elfedit
    ln -sf riscv64-buildroot-linux-gnu-c++ riscv64-linux-gnu-c++
    ln -sf riscv64-buildroot-linux-gnu-cpp riscv64-linux-gnu-cpp
    ln -sf riscv64-buildroot-linux-gnu-gcov riscv64-linux-gnu-gcov
    ln -sf riscv64-buildroot-linux-gnu-gcov-dump riscv64-linux-gnu-gcov-dump
    ln -sf riscv64-buildroot-linux-gnu-gcov-tool riscv64-linux-gnu-gcov-tool
    ln -sf riscv64-buildroot-linux-gnu-cc riscv64-linux-gnu-cc
    ln -sf riscv64-buildroot-linux-gnu-c++ riscv64-linux-gnu-c++
    
    popd
}

# Set up PATH with RISC-V toolchain
export PATH="$(pwd)/riscv64-toolchain/bin:$PATH"

# Verify toolchain
echo "Using RISC-V toolchain:"
riscv64-linux-gnu-gcc --version | head -1

mkdir -p .build
pushd .build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=OFF
make -j$(nproc)
popd