set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR "riscv64")
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_C_COMPILER "zig" cc -target riscv64-linux-musl)
set(CMAKE_CXX_COMPILER "zig" c++ -target riscv64-linux-musl)
set(CMAKE_C_FLAGS_INIT "-ffreestanding")
set(CMAKE_CXX_FLAGS_INIT "-ffreestanding")

# Prevent linking zig's libc to avoid duplicates with sandbox API, but link libcxx for C++ features
set(CMAKE_EXE_LINKER_FLAGS "-nodefaultlibs -lc++")

if (CMAKE_HOST_WIN32)
	# Windows: Disable .d files
	set(CMAKE_C_LINKER_DEPFILE_SUPPORTED FALSE)
	set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED FALSE)
	# Windows: Work-around for zig ar and zig ranlib
	set(CMAKE_AR "${CMAKE_CURRENT_LIST_DIR}/zig-ar.cmd")
	set(CMAKE_RANLIB "${CMAKE_CURRENT_LIST_DIR}/zig-ranlib.cmd")
endif()
