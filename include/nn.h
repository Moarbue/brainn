#include "../config.h"
#include "mat.h"
#include "vec.h"
#include "vec_mat.h"
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
bfloat nn_loss(NN nn, Mat training_inputs, Mat training_outputs);
void   nn_train(NN nn, Mat ti, Mat to, bsize batch_size, bsize epochs, bfloat lr, int report_loss);
void   nn_ptrain(NN *nn, Mat ti, Mat to, bsize batch_size, bsize epochs, bfloat lr, bsize nthreads, int report_loss);
void   nn_free(NN nn);

// io functions
void nn_save(const char *filename, NN nn);
NN   nn_load(const char *filename);

// config functions
void nn_set_activation_function(NN *nn, hidden_activation hf, dhidden_activation dhf, 
                                        output_activation of, doutput_activation dof);

void nn_set_loss_function(NN *nn, loss_function C, dloss_function dC);

void nn_get_arch(NN nn, bsize *arch[], bsize *layers);

void nn_print_intern(NN nn, const char *name);

#endif // _BRAINN_NN_H_