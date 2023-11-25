#include "../config.h"
#include "mat.h"
#include "vec.h"
#include "activation.h"
#include "loss.h"

#ifndef _BRAINN_NN_H_
#define _BRAINN_NN_H_

#define nn_input(nn) (nn).a[0]
#define nn_output(nn) (nn).a[(nn).l]
#define nn_print(nn)  nn_print_intern(nn, #nn)

typedef struct {
    bsize l;    // layer count (input exclusive)
    Vec *a;     // neuron after  activation
    Vec *z;     // neuron before activation
    Vec *b;     // biases
    Mat *w;     // weights

    Vec *da;    // derivative of Loss function in respect to neuron output
    Vec *gb;    // gradient of biases
    Mat *gw;    // gradient of weights
    bsize *gc;  // backprop iteration count

    hidden_activation  *hf;
    dhidden_activation *dhf;
    output_activation  *of;
    doutput_activation *dof;
    loss_function  *C;
    dloss_function *dC;
} NN;

NN     nn_alloc(bsize *arch, bsize layers);
void   nn_init(NN nn, bfloat min, bfloat max);
Vec    nn_forward(NN nn, Vec input);
void   nn_backpropagate(NN nn, Vec output);
void   nn_evolve(NN nn, bfloat lr);
bfloat nn_loss(NN nn, Vec *training_inputs, Vec *training_outputs, bsize samples);
void   nn_free(NN nn);

void nn_print_intern(NN nn, const char *name);

#endif // _BRAINN_NN_H_