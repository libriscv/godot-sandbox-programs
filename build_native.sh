#!/bin/bash
# Native x86_64 Build Script
# Builds the scenetree_passthrough program for the host architecture

echo "=== Native x86_64 Build ==="
echo "Building for: $(uname -m)"

# Use native toolchain
TOOLCHAIN=$(pwd)/cmake/native-toolchain.cmake

echo "Toolchain: $TOOLCHAIN"

# Clean previous build (optional - comment out to keep)
# rm -rf .build_native

mkdir -p .build_native
pushd .build_native

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRIPPED=OFF \
    -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN \
    -DCMAKE_CROSSCOMPILING=OFF

if [ $? -eq 0 ]; then
    make -j$(nproc)
else
    echo "CMake configuration failed"
    exit 1
fi

popd
echo "=== Build complete ==="
echo "Output: .build_native/bin/scenetree_passthrough"
