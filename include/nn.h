#include <stddef.h>

#include "vec.h"
#include "mat.h"
#include "activation.h"
#include "loss.h"

#ifndef BRAINN_NN_H_
#define BRAINN_NN_H_

#define nn_input(nn)  (nn).a[0]
#define nn_output(nn) (nn).a[(nn).l]
#define nn_print(nn)  nn_print_intern(nn, #nn)
typedef struct {
    size_t l;   // layer count (input layer exclusive)
    Vec *a;     // activations
    Vec *b;     // biases
    Mat *w;     // weights

    hidden_activation_function *haf;
    output_activation_function *oaf;
    loss_function *lf;
    dloss_function *dlf;
} NN;

NN    nn_alloc(size_t *arch, size_t layers);
void  nn_init(NN nn, float min, float max);
void  nn_fill(NN nn, size_t val);
Vec   nn_forward(NN nn);
float nn_loss(NN nn, Mat training_inputs, Mat expected_outputs);
void  nn_free(NN nn);

void nn_set_activation_function(NN *nn, hidden_activation_function *haf, output_activation_function *oaf);
void nn_set_loss_functions(NN *nn, loss_function *lf, dloss_function *dlf);

void nn_print_intern(NN nn, const char *name);

#endif // BRAINN_NN_H_