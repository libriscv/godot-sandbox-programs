CC="riscv64-unknown-elf-gcc"
CXX="riscv64-unknown-elf-g++"
TOOLCHAIN=$PWD/ext/godot-sandbox/programs/cpp/cmake/toolchain.cmake

mkdir -p .embedded
pushd .embedded
cmake -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=ON
make -j
popd
