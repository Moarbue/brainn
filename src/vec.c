#include "../include/vec.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

Vec  vec_alloc(size_t elements)
{
    Vec v;

    v.c = elements;
    v.e = (float *) malloc(sizeof (*v.e) * elements);

    assert(v.e != NULL && "Failed to allocate memory for vector!");

    return v;
}

void vec_fill(Vec v, float val)
{
    assert(v.e != NULL && "No memory for elements allocated!");

    for (size_t i = 0; i < v.c; i++) {
        vec_el(v, i) = val;
    } 
}

void vec_rand(Vec v, float min, float max)
{
    assert(v.e != NULL && "No memory for elements allocated!");

    for (size_t i = 0; i < v.c; i++) {
        vec_el(v, i) = ((float) rand() / (float) RAND_MAX) * (max - min) + min;
    }
}

void vec_copy(Vec dst, Vec src)
{
    assert(dst.e != NULL  && "No memory for elements of dst allocated!");
    assert(src.e != NULL  && "No memory for elements of src allocated!");
    assert(dst.c == src.c && "Destination size doesn't match source size");

    for (size_t i = 0; i < dst.c; i++) {
        vec_el(dst, i) = vec_el(src, i);
    }
}

void vec_sum(Vec dst, Vec v)
{
    assert(dst.e != NULL && "No memory for elements of dst allocated!");
    assert(v.e   != NULL && "No memory for elements of v allocated!");
    assert(dst.c == v.c  && "Destination size doesn't match source size");

    for (size_t i = 0; i < dst.c; i++) {
        vec_el(dst, i) += vec_el(v, i);
    }
}

void vec_free(Vec v)
{
    free(v.e);
}


void vec_print_intern(Vec v, const char *name, size_t pad)
{
    assert(v.e != NULL && "No memory for elements allocated!");

    printf("%*s%s = [\n", (int) pad, "", name);
    for (size_t c = 0; c < v.c; c++) {
        printf("%*s    %.3f\n", (int) pad, "", v.e[c]);
    }
    printf("%*s]\n", (int) pad, "");
}
