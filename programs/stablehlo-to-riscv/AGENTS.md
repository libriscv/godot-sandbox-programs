# StableHLO to RISC-V ELF64 - Agent Documentation

This document provides reference information for AI agents working on the `stablehlo-to-riscv` program.

## Overview

The `stablehlo-to-riscv` program compiles StableHLO MLIR code directly to RISC-V ELF64 binaries. It runs inside the Godot sandbox (a RISC-V emulator) and must work entirely headlessly without external process execution.

## Key Architecture Decisions

### Headless Build Requirement

**Problem**: The program runs inside a RISC-V sandbox emulator where external process execution (`std::system()`, `fork()`, `exec()`) is not available.

**Solution**: Generate minimal ELF64 executables directly from LLVM object files using LLVM's Object library, eliminating the need for external linkers.

### Implementation Approach

1. **StableHLO → LLVM IR**: Use MLIR to lower StableHLO through the dialect pipeline
2. **LLVM IR → Object File**: Use LLVM's codegen to emit RISC-V object files
3. **Object File → ELF64**: Read the object file using LLVM's Object library and construct a minimal ELF64 executable directly

## File Structure

```
programs/stablehlo-to-riscv/
├── main.cpp              # Godot sandbox entry point, exposes API functions
├── mlir_compiler.h       # MLIRCompiler class interface
├── mlir_compiler.cpp     # Implementation: MLIR lowering + ELF generation
├── CMakeLists.txt        # Build configuration (LLVM/MLIR/StableHLO setup)
├── test_fixtures/        # Test StableHLO MLIR files
│   ├── test_simple.mlir
│   ├── test_add.mlir
│   ├── test_multiply.mlir
│   ├── test_assign.mlir
│   ├── test_if.mlir
│   └── e2e_test.mlir
└── AGENTS.md             # This file
```

## Build Dependencies

The build system automatically downloads and builds:

- **LLVM 18.1.0** with MLIR and RISC-V target support
- **StableHLO v1.0.0** for StableHLO dialect support

These are built as ExternalProjects during the CMake configuration phase.

## Key Code Components

### `mlir_compiler.cpp`

#### `lower_to_llvm()`

Lowers StableHLO MLIR to LLVM IR through the dialect pipeline:

- StableHLO → HLO → Linalg → Affine → Standard → LLVM

#### `compile_llvm_to_riscv_elf64()`

1. Parses LLVM IR
2. Creates RISC-V target machine
3. Emits object file using LLVM codegen
4. Calls `create_minimal_elf_executable()` to generate final ELF

#### `create_minimal_elf_executable()`

**Critical function** - Generates ELF64 executable without external linker:

1. Reads object file using `llvm::object::ObjectFile`
2. Extracts `.text` section and entry point symbol
3. Constructs minimal ELF64 structure:
   - ELF header (64 bytes)
   - Program header (PT_LOAD segment)
   - Code section data
4. Returns complete ELF64 binary

### ELF64 Structure

The generated ELF64 has:

- **ELF Header**: Standard 64-bit ELF header with RISC-V machine type (0xF3)
- **Program Header**: Single PT_LOAD segment for code (readable + executable)
- **Code Section**: Extracted from object file's `.text` section
- **Entry Point**: Found from `_start`, `main`, or `_main` symbol, defaults to 0x10000

## Testing Test Fixtures

All test fixtures in `test_fixtures/` are validated to be translatable with StableHLO tools:

```bash
cd programs/stablehlo-to-riscv/test_fixtures
stablehlo-opt test_simple.mlir  # Should succeed
```

### Test Fixture Requirements

1. **Float literals**: Must use decimal notation (e.g., `42.0` not `42`)
2. **Supported operations**:
   - ✅ `stablehlo.constant`, `stablehlo.add`, `stablehlo.multiply`
   - ✅ `stablehlo.select` (for conditionals)
   - ✅ `stablehlo.compare` (for comparisons)
   - ❌ `stablehlo.copy` (not available in this version)
   - ❌ `stablehlo.if` (use `stablehlo.select` instead)

### Example Valid Test Fixture

```mlir
module {
  func.func @test_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %v0 = stablehlo.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    stablehlo.return %v0 : tensor<f32>
  }
}
```

## Common Issues and Solutions

### Issue: "ELFObjectFileBase not found"

**Solution**: Use template-based approach:

```cpp
using ELF64LE = llvm::object::ELFObjectFile<llvm::object::ELF64LEFile>;
auto elf_obj = llvm::dyn_cast<ELF64LE>(bin);
```

### Issue: "Cannot execute external linker"

**Solution**: This is expected - the sandbox cannot execute external processes. Use `create_minimal_elf_executable()` instead.

### Issue: "No .text section found"

**Solution**: Ensure the LLVM IR actually generates code. Check that:

- Functions are not optimized away
- Target machine is correctly configured
- Object file generation succeeds

### Issue: Test fixtures fail with "unexpected decimal integer literal"

**Solution**: Use float literals with decimal point: `42.0` instead of `42`

## API Functions Exposed to Godot

The program exposes these functions via `ADD_API_FUNCTION`:

1. **`compile_stablehlo_to_riscv_elf64(String stablehlo_content)`**

   - Takes StableHLO MLIR text
   - Returns `PackedByteArray` containing RISC-V ELF64 binary

2. **`is_mlir_available()`**

   - Returns `bool` indicating if MLIR is available

3. **`get_mlir_version()`**
   - Returns `String` with MLIR version info

## Build Configuration Notes

### CMakeLists.txt Key Points

- **LLVM_ENABLE_PROJECTS**: Set to `"mlir"` (lld not needed - we generate ELF directly)
- **LLVM_TARGETS_TO_BUILD**: Includes `"RISCV"` for RISC-V codegen
- **RISC-V flags removed**: The `add_ci_program` macro adds RISC-V flags, but we're building for host, so they're filtered out
- **External dependencies**: LLVM and StableHLO are built as ExternalProjects before the main target

## Future Improvements

1. **Better entry point detection**: Currently searches for `_start`, `main`, or `_main` - could be more robust
2. **Multiple sections**: Currently only handles `.text` - could support `.data`, `.rodata`, etc.
3. **Relocation support**: Currently generates static executables - dynamic linking would require more work
4. **Error handling**: More detailed error messages for debugging

## References

- LLVM Object Library: https://llvm.org/docs/ProgrammersManual.html#object-file-handling
- ELF64 Specification: https://en.wikipedia.org/wiki/Executable_and_Linkable_Format
- StableHLO Dialect: https://github.com/openxla/stablehlo
- MLIR Dialect Lowering: https://mlir.llvm.org/docs/Dialects/
