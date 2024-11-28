#include <string.h>
#include <stdio.h>
struct AsmjitResult {
	unsigned char *data;
	size_t size;
};
extern struct AsmjitResult assemble_to(char *input, size_t size);

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <input>\n", argv[0]);
		return 1;
	}
	struct AsmjitResult result = assemble_to(argv[1], strlen(argv[1]));

	register char  *a0 asm("a0") = result.data;
	register size_t a1 asm("a1") = result.size;
	register long a7 asm("a7") = 93; // __NR_exit_group

	asm volatile("ecall" : : "r"(a0), "m"(*a0), "r"(a1), "r"(a7) : "memory");
}
