#include <stddef.h>

#ifndef BRAINN_VEC_H_
#define BRAINN_VEC_H_

#define vec_el(v, i) (v).e[(i)]
typedef struct {
    size_t c;   // element count
    float *e;   // elements
} Vec;

Vec  vec_alloc(size_t elements);
void vec_fill(Vec v, float val);
void vec_rand(Vec v, float min, float max);
void vec_copy(Vec dst, Vec src);
void vec_sum(Vec dst, Vec v);
void vec_free(Vec v);

#endif // BRAINN_VEC_H_