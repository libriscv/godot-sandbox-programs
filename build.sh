export CC="riscv64-linux-gnu-gcc-12"
export CXX="riscv64-linux-gnu-g++-12"

mkdir -p .build
pushd .build
cmake -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=ON
make -j
popd
