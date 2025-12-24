# GDScript compiler

## Get program

Run this GDScript:

```cpp
if !FileAccess.file_exists("res://gdscript.elf"):
	var buffer = Sandbox.download_program("gdscript")
	fa = FileAccess.open("res://gdscript.elf", FileAccess.WRITE)
	fa.store_buffer(buffer)
	fa.close()

var compiler = Node.new()
compiler.set_script(load("res://gdscript.elf"))
compiler.set_restrictions(true)
```

## Usage

Compiles basic GDScript to a buffer that is loadable as a Sandbox program.

```py
var compiled_elf = compiler.vmcall("compile_to_elf", """
func test():
	var matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	var sum = 0
	for i in range(3):
		for j in range(3):
			sum = sum + matrix[i][j]
	return sum
""")

var s = Sandbox.new()
s.restrictions = true
s.load_buffer(compiled_elf)
```

Calculates the sum:

```py
print(s.vmcallv("test"))
```
