// General configuration for the library

#ifndef _BRAINN_CONFIG_H
#define _BRAINN_CONFIG_H

// for size_t
#include <inttypes.h>
// rand() and malloc()
#include <stdlib.h>
// printf()
#include <stdio.h>

#define BALLOC malloc
#define BREALLOC realloc
#define BFREE  free

#define PANIC(msg, ...) do { printf(msg "\n", ##__VA_ARGS__); exit(EXIT_FAILURE); } while(0)

// type of floating precision number
typedef float bfloat;
// type of unsigned integer
typedef size_t bsize;

// random function
static inline bfloat brand(bfloat min, bfloat max)
{
    return ((bfloat) rand() / (bfloat) RAND_MAX) * (max - min) + min;
}

#endif // _BRAINN_CONFIG_H