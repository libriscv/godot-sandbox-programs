# MLIR/StableHLO Test Fixtures

This directory contains MLIR (StableHLO) test fixtures generated from GDScript test cases in the `gdscript_elf` module.

## Files

- `test_simple.mlir` - Simple function returning a constant
- `test_add.mlir` - Addition operation
- `test_multiply.mlir` - Multiplication operation  
- `test_if.mlir` - Conditional logic
- `test_assign.mlir` - Variable assignments
- `e2e_test.mlir` - End-to-end test with multiple operations

## Source

These fixtures are based on test cases from:
- `modules/gdscript_elf/tests/test_gdscript_elf_e2e.h`

## Format

All fixtures use the StableHLO dialect in MLIR text format:
- Functions use `func.func @function_name` syntax
- Operations use `stablehlo.*` operations
- All values are `tensor<f32>` type
- Constants use `stablehlo.constant dense<value>`

## Usage

These fixtures can be used to test the StableHLO to RISC-V compilation pipeline in the `godot-sandbox-programs` project.
