# Godot Sandbox Programs

This repository can be used as a starting point for automatically building and maintaining Sandbox programs with CMake.

All programs are built in CI using a modern C/C++ RISC-V toolchain, with access to the Godot Sandbox C++ API, and automatically zipped and uploaded to a Draft release. Fork this repository and build your own Sandbox programs in a maintainable way!


## Usage

If you are using Godot Sandbox, each program built by this repository can be downloaded by a Sandbox node using `Sandbox.download_program(name)`. See the READMEs of each program for how to use.

## Example programs

### Hello World

The [Hello World example](/programs/hello-world) is a minimal project that can be used as a starting point for anyone who wants to write Sandbox programs using modern C++.

### Asm JIT example

A [RISC-V assembler](/programs/asm) is embedded into a Sandbox program. It will assemble RISC-V and return a callable.

### LuaJit example

[LuaJit is embedded](/programs/luajit) into a Sandbox program. It can be used to run JIT-compiled Lua at run-time.

### libtcc example

[Libtcc is embedded](/programs/libtcc) into a Sandbox program. It can be used to compile and run C-code at run-time.

### Mir example

[Mir is embedded](/programs/mir) into a Sandbox program. It can be used to compile and run C-code at run-time.

### Robust Skin Weights Transfer via Weight Inpainting

![Teaser](https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/teaser.jpg)

A re-implementation of [Robust Skin Weights Transfer via Weight Inpainting](https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/index.html). If you use this code for an academic publication, cite it as:

```bib
@inproceedings{abdrashitov2023robust,
author = {Abdrashitov, Rinat and Raichstat, Kim and Monsen, Jared and Hill, David},
title = {Robust Skin Weights Transfer via Weight Inpainting},
year = {2023},
isbn = {9798400703140},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3610543.3626180},
doi = {10.1145/3610543.3626180},
booktitle = {SIGGRAPH Asia 2023 Technical Communications},
articleno = {25},
numpages = {4},
location = {<conf-loc>, <city>Sydney</city>, <state>NSW</state>, <country>Australia</country>, </conf-loc>},
series = {SA '23}
}
```
