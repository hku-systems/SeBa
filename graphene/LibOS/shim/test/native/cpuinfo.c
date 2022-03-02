#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    FILE* cpuinfo = fopen("/proc/cpuinfo", "rb");
    char* arg     = 0;
    size_t size   = 0;

    if (!cpuinfo)
        return errno;

    while (getdelim(&arg, &size, 0, cpuinfo) != -1) {
        puts(arg);
    }

    free(arg);
    fclose(cpuinfo);
    return 0;
}
