# MLIR Test Fixtures

This directory contains MLIR/StableHLO test fixtures generated from GDScript test cases.

## Test Cases

### test_simple.mlir
```gdscript
func test_simple() -> int:
	return 42
```

### test_add.mlir
```gdscript
func test_add(a: int, b: int) -> int:
	return a + b
```

### test_multiply.mlir
```gdscript
func test_multiply(x: int, y: int) -> int:
	return x * y
```

### test_if.mlir
```gdscript
func test_if(x: int) -> int:
	if x > 10:
		return 100
	return 0
```

### test_assign.mlir
```gdscript
func test_assign() -> int:
	var x = 5
	var y = 10
	return x + y
```

### e2e_test.mlir
```gdscript
func e2e_test(a: int, b: int) -> int:
	var result = a + b
	if result > 100:
		return 100
	return result
```

## Generation

These MLIR files are generated from GDScript functions using `GDScriptToStableHLO::convert_function_to_stablehlo_text()`.
