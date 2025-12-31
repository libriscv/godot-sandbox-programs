TOOLCHAIN=$PWD/cmake/riscv-toolchain.cmake

mkdir -p build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=ON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN
make -j8
popd

