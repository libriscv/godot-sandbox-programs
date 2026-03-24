#!/bin/bash
# RISC-V Build Script using Bootlin Pre-Built Toolchain

# Download Bootlin toolchain (if not already present):
# wget https://toolchains.bootlin.com/downloads/releases/toolchains/riscv64/tarballs/riscv64--glibc--stable-2023.08-1.tar.bz2
# tar -xjf riscv64--glibc--stable-2023.08-1.tar.bz2

# Add to PATH (adjust path as needed):
# export PATH=$PWD/riscv64--glibc--stable-2023.08-1/bin:$PATH

# Verify toolchain:
# riscv64-linux-gnu-gcc --version

# Add RISC-V toolchain to PATH
export PATH=$PWD/riscv64-lp64d--glibc--stable-2025.08-1/bin:$PATH

TOOLCHAIN=$PWD/cmake/riscv-toolchain.cmake

mkdir -p build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=OFF -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN
make -j8
popd



