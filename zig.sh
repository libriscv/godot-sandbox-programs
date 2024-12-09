export CC="zig;cc;-target riscv64-linux-musl"
export CXX="zig;c++;-target riscv64-linux-musl"

mkdir -p .zig
pushd .zig
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=ON -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
make -j8
popd
