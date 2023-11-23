#include "../include/vec.h"

Vec vec_alloc(bsize s)
{
    Vec v;

    v.s = s;
    v.e = (bfloat*) BALLOC(v.s * sizeof (bfloat));

    if (v.e == NULL) PANIC("vec_alloc(): Failed to allocate memory!");

    return v;
}

void vec_fill(Vec v, bfloat value)
{
    for (bsize i = 0; i < v.s; i++) {
        vec_el(v, i) = value;
    }
}

void vec_copy(Vec dst, Vec src)
{
    if (dst.s != src.s) PANIC("vec_copy(): Destination size differs from source size!");

    for (bsize i = 0; i < src.s; i++) {
        vec_el(dst, i) = vec_el(src, i);
    }
}

void vec_sum(Vec dst, Vec v)
{
    if (dst.s != v.s) PANIC("vec_sum(): Destination size differs from source size!");

    for (bsize i = 0; i < dst.s; i++) {
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
    for (bsize i = 0; i < v.s; i++) {
        printf("%*s    %.3f\n", (int) pad, "", vec_el(v, i));
    }
    printf("%*s]\n", (int) pad, "");
}