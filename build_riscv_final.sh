#!/bin/bash
set -e

# RISC-V Build Script using Bootlin Pre-Built Toolchain
export PATH=$PWD/riscv64-lp64d--glibc--stable-2025.08-1/bin:$PATH

TOOLCHAIN=$PWD/cmake/riscv-toolchain.cmake

echo "Building with RISC-V toolchain..."
mkdir -p build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=OFF -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN
time make -j$(nproc)
popd

echo "Build complete!"
echo "RISC-V ELF files:"
file build/bin/*
