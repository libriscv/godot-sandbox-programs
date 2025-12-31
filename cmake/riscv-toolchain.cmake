set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR "riscv64")
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

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

if (CMAKE_HOST_WIN32)
	# Windows: Disable .d files (dependency files)
	set(CMAKE_C_LINKER_DEPFILE_SUPPORTED FALSE)
	set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED FALSE)
endif()

