#include "../config.h"
#include "mat.h"
#include "vec.h"
#include "vec_mat.h"

#ifndef _BRAINN_NN_H_
#define _BRAINN_NN_H_

#define nn_input(nn) (nn).a[0]
#define nn_output(nn) (nn).a[(nn).l]
#define nn_print(nn)  nn_print_intern(nn, #nn)

typedef struct {
    bsize l;    // layer count (input exclusive)
    Vec *a;     // activations
    Vec *b;     // biases
    Mat *w;     // weights
} NN;

NN   nn_alloc(bsize *arch, bsize layers);
void nn_init(NN nn, bfloat min, bfloat max);
Vec  nn_forward(NN nn, Vec input);
void nn_free(NN nn);

void nn_print_intern(NN nn, const char *name);

#endif // _BRAINN_NN_H_