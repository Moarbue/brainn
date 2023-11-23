#include "../config.h"

#ifndef _BRAINN_VEC_H
#define _BRAINN_VEC_H

#define vec_el(v, i) (v).e[(i)]
#define vec_print(v, pad) vec_print_intern(v, #v, pad)

typedef struct {
    bsize s;    // size
    bfloat *e;
} Vec;

Vec  vec_alloc(bsize s);
void vec_fill(Vec v, bfloat value);
void vec_copy(Vec dst, Vec src);
void vec_sum(Vec dst, Vec v);
void vec_free(Vec v);

void vec_print_intern(Vec v, const char *name, bsize pad);

#endif // _BRAINN_VEC_H
