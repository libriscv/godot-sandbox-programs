# Hello World

## Get program

Run this GDScript:

```cpp
if !FileAccess.file_exists("res://hello_world.elf"):
	var buffer = Sandbox.download_program("hello_world")
	fa = FileAccess.open("res://hello_world.elf", FileAccess.WRITE)
	fa.store_buffer(buffer)
	fa.close()

var hello = Node.new()
hello.set_script(load("res://hello_world.elf"))
```

## Usage

Prints "Hello, world!"

```cpp
print(hello.hello_world())
```

Calculates the nth Fibonacci number:

```cpp
print(hello.fibonacci(256))
```
