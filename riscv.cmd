set HERE=%cd%
mkdir build
pushd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=ON -DCMAKE_TOOLCHAIN_FILE=%HERE%\cmake\riscv-toolchain.cmake
cmake --build .
popd

