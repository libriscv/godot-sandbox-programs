# Godot Sandbox Programs

This repository can be used as a starting point for automatically building and maintaining Sandbox programs with CMake.

All programs are built in CI using a modern C/C++ RISC-V toolchain, with access to the Godot Sandbox C++ API, and automatically zipped and uploaded to a Draft release. Fork this repository and build your own Sandbox programs in a maintainable way!


## Example programs

### Hello World

The [Hello World example](/hello-world) is a minimal project that can be used as a starting point for anyone who wants to write Sandbox programs using modern C++.

### Asm JIT example

A [RISC-V assembler](/asm) is embedded into a Sandbox program. It will assemble RISC-V and return a callable.

### LuaJit example

[LuaJit is embedded](/luajit) into a Sandbox program. It can be used to run JIT-compiled Lua at run-time.

### libtcc example

[Libtcc is embedded](/libtcc) into a Sandbox program. It can be used to compile and run C-code at run-time.

### Mir example

[Mir is embedded](/mir) into a Sandbox program. It can be used to compile and run C-code at run-time.
