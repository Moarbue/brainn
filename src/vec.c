#include "../include/vec.h"

Vec vec_alloc(bsize c)
{
    Vec v;

    v.c = c;
    v.s = 1;
    v.e = (bfloat*) BALLOC(v.c * sizeof (bfloat));

    if (v.e == NULL) PANIC("vec_alloc(): Failed to allocate memory!");

    return v;
}

void vec_fill(Vec v, bfloat value)
{
    for (bsize i = 0; i < v.c; i++) {
        vec_el(v, i) = value;
    }
}

void vec_copy(Vec dst, Vec src)
{
    if (dst.c != src.c) PANIC("vec_copy(): Destination size differs from source size!");

    for (bsize i = 0; i < src.c; i++) {
        vec_el(dst, i) = vec_el(src, i);
    }
}

void vec_sum(Vec dst, Vec v)
{
    if (dst.c != v.c) PANIC("vec_sum(): Destination size differs from source size!");

    for (bsize i = 0; i < dst.c; i++) {
        vec_el(dst, i) += vec_el(v, i);
    }
}

void vec_free(Vec v)
{
    BFREE(v.e);
}


void vec_print_intern(Vec v, const char *name, bsize pad)
{
    printf("%*s%s = [\n", (int) pad, "", name);
    for (bsize i = 0; i < v.c; i++) {
        printf("%*s    %.3f\n", (int) pad, "", vec_el(v, i));
    }
    printf("%*s]\n", (int) pad, "");
}