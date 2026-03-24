set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR "riscv64")
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Fedora Setup Notes:
#   To use Bootlin pre-built toolchain on Fedora:
#   1. Remove conflicting packages: sudo dnf remove gcc-c++-riscv64-linux-gnu gcc-riscv64-linux-gnu
#   2. Download: `curl -L https://toolchains.bootlin.com/downloads/releases/toolchains/riscv64-lp64d/tarballs/riscv64-lp64d--glibc--stable-2025.08-1.tar.xz -O riscv64-lp64d--glibc--stable-2025.08-1.tar.xz`
#   3. Extract: `tar -xf riscv64-lp64d--glibc--stable-2025.08-1.tar.xz`
#   4. Add to PATH: `export PATH=$PWD/riscv64-lp64d--glibc--stable-2025.08-1/bin:$PATH`
#   5. Remove `rm -rf .build`.
#   6. Build: `./build.sh`

# Find RISC-V native compilers in PATH
# Try common names: riscv64-linux-gnu-gcc, riscv64-unknown-linux-gnu-gcc, etc.
find_program(CMAKE_C_COMPILER
	NAMES
		riscv64-linux-gnu-gcc
		riscv64-unknown-linux-gnu-gcc
		riscv64-linux-gnu-gcc-14
		riscv64-linux-gnu-gcc-13
		riscv64-linux-gnu-gcc-12
		riscv64-linux-gnu-gcc-11
	PATHS
		ENV PATH
)

find_program(CMAKE_CXX_COMPILER
	NAMES
		riscv64-linux-gnu-g++
		riscv64-unknown-linux-gnu-g++
		riscv64-linux-gnu-g++-14
		riscv64-linux-gnu-g++-13
		riscv64-linux-gnu-g++-12
		riscv64-linux-gnu-g++-11
	PATHS
		ENV PATH
)

if (NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
	message(FATAL_ERROR
		"RISC-V native compilers not found in PATH.\n"
		"Please install the RISC-V GNU toolchain:\n"
		"  Windows (MSYS2): pacman -S mingw-w64-riscv64-toolchain\n"
		"  Windows (Scoop): scoop install riscv-gnu-toolchain (if available)\n"
		"  macOS: brew tap riscv-software-src/riscv && brew install riscv-tools\n"
		"  Linux: apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu"
	)
endif()

message(STATUS "Found RISC-V C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Found RISC-V C++ compiler: ${CMAKE_CXX_COMPILER}")

# Set RISC-V architecture flags to rv64imafdc with lp64d ABI
# This ensures the compiler defines the required preprocessor macros:
# __riscv_xlen=64, __riscv_flen=64, __riscv_float_abi_double,
# __riscv_mul, __riscv_div, __riscv_compressed
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64imafdc -mabi=lp64d")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64imafdc -mabi=lp64d")

if (CMAKE_HOST_WIN32)
	# Windows: Disable .d files (dependency files)
	set(CMAKE_C_LINKER_DEPFILE_SUPPORTED FALSE)
	set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED FALSE)
endif()

