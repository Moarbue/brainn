#include <stddef.h>
#include "mat.h"

#ifndef BRAINN_VEC_H_
#define BRAINN_VEC_H_

#define vec_el(v, i) (v).e[(i) + (i) * (v).s]
#define vec_print(v, pad) vec_print_intern(v, #v, pad)
typedef struct {
    size_t c;   // element count
    size_t s;   // stride
    float *e;   // elements
} Vec;

Vec   vec_alloc(size_t elements);
void  vec_fill(Vec v, float val);
void  vec_rand(Vec v, float min, float max);
void  vec_copy(Vec dst, Vec src);
void  vec_sum(Vec dst, Vec v);
float vec_max(Vec v);
void  vec_free(Vec v);

void vec_print_intern(Vec v, const char *name, size_t pad);

#endif // BRAINN_VEC_H_