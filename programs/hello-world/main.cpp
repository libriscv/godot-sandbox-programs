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

SANDBOX_API({
	.name = "hello_world",
	.address = (void *)hello_world,
	.description = "Returns the string 'Hello, world!'",
	.return_type = "String",
	.arguments = "",
});
