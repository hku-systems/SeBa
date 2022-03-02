#include <stdio.h>
#include <sys/mman.h>

int main() {
        int n = 0;
        int PAGES = 1 * 64 * 4096; // 1GB MEM
        int SIZE = PAGES * 4096;
        char* arr = (int*) mmap (0, SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        printf("addr 0x%lx -> 0x%lx\n", &arr[0], &arr[SIZE - 1]);
	fprintf(stderr,"start exec\n");
        for (int j = 0;j < 1;j++) {
                for (int i = 0;i < PAGES;i++) {
			arr[i * 4096 + 137] = j + i;
                        n += (int)arr[i * 4096 + 137];
                }
        }
        fprintf(stderr, "result is %ld\n", n);
}
