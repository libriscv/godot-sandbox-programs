#include <api.hpp>
static Variant test_memalign();

static Variant hello_world() {
	return "Hello, world!";
}
static Variant print_string(String str) {
	printf("String: %s\n", str.utf8().c_str());
	fflush(stdout);
	return Nil;
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

static int meaning_of_life = 42;
int main() {
	print("Hello, world!");

	// The entire Godot API is available
	Sandbox sandbox = get_node<Sandbox>();
	print(sandbox.is_binary_translated()
		? "The current program is accelerated by binary translation."
		: "The current program is running in interpreter mode.");

	// Add public API
	ADD_API_FUNCTION(hello_world, "String", "", "Returns the string 'Hello, world!'");
	ADD_API_FUNCTION(print_string, "void", "String str", "Prints a string to the console");
	ADD_API_FUNCTION(fibonacci, "long", "int n", "Calculates the nth Fibonacci number");
	ADD_API_FUNCTION(test_memalign, "void", "", "Tests memory alignment");

	// Add a property
	add_property("meaning_of_life", Variant::Type::INT, 42,
		[]() -> Variant { return meaning_of_life; },
		[](Variant value) -> Variant { meaning_of_life = value; print("Set to: ", meaning_of_life); return Nil; });

	halt();
}

Variant test_memalign() {
	struct alignas(32) Test {
		int a;
		int b;
		int c;
		int d;
	};
	std::vector<Test*> test;
	std::vector<char*> chars;
	for (int i = 0; i < 100; i++) {
		Test *t1 = new Test;
		Test *t2 = new Test;
		printf("Test: %p, %p\n", t1, t2);
		test.push_back(t2);
		char *c = new char[1];
		chars.push_back(c);
		delete t1;
	}
	for (auto t : test) {
		delete t;
	}
	for (auto c : chars) {
		delete[] c;
	}
	fflush(stdout);
	return Nil;
}
