mkdir -p .build
pushd .build
cmake -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX .. -DCMAKE_BUILD_TYPE=Release -DSTRIPPED=ON
make -j
popd
