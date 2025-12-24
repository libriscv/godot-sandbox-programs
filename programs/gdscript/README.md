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
	print("Loading sandboxed GDScript!")
	var gdscript = Node.new()
	gdscript.set_script(load("res://gdscript.elf"))
	var compiled_elf = gdscript.compile("""
func sum(n):
	var total = 0
	var i = 0
	while i <= n:
		total += i
		i += 1
	return total
	""")

	var s = Sandbox.new()
	s.restrictions = true
	s.load_buffer(compiled_elf)
```

Calculate the sum:

```py
print("Sum: ", s.vmcallv("sum", 5))
```
