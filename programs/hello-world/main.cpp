#include <api.hpp>

int main() {
	print("Hello, world!\n");

	// Create a new Godot Sandbox node.
	Sandbox sandbox = ClassDB::instantiate("Sandbox", "my_sandbox");
	// Delete it next frame.
	sandbox.queue_free();

	halt();
}

static Variant hello_world() {
	return "Hello, world!";
}

static long fib(long n, long acc, long prev)
{
	if (n == 0)
		return acc;
	else
		return fib(n - 1, prev + acc, acc);
}

static Variant fibonacci(int n) {
	return fib(n, 0, 1);
}

SANDBOX_API({
	.name = "hello_world",
	.address = (void *)hello_world,
	.description = "Returns the string 'Hello, world!'",
	.return_type = "String",
	.arguments = "",
}, {
	.name = "fibonacci",
	.address = (void *)fibonacci,
	.description = "Calculates the nth Fibonacci number",
	.return_type = "long",
	.arguments = "int n",
});
