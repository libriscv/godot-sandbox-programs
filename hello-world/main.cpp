#include <api.hpp>

int main() {
	print("Hello, world!\n");
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
