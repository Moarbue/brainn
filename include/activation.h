#include "../config.h"
#include "vec.h"

#ifndef _BRAINN_ACTIVATION_H_
#define _BRAINN_ACTIVATION_H_

typedef bfloat (activation_function)(bfloat);
typedef bfloat (hidden_activation) (bfloat);
typedef bfloat (dhidden_activation)(bfloat);
typedef bfloat (output_activation) (bfloat);
typedef bfloat (doutput_activation)(bfloat);

bfloat Sigmoid(bfloat z);
bfloat Tanh(bfloat z);
bfloat ReLU(bfloat z);
bfloat Heaviside(bfloat z);
bfloat GELU(bfloat z);
bfloat Softplus(bfloat z);
bfloat lReLU(bfloat z);

bfloat dSigmoid(bfloat z);
bfloat dTanh(bfloat z);
bfloat dReLU(bfloat z);
bfloat dHeaviside(bfloat z);
bfloat dGELU(bfloat z);
bfloat dSoftplus(bfloat z);
bfloat dlReLU(bfloat z);

void vec_activate(Vec v, activation_function *af);

#endif // _BRAINN_ACTIVATION_H_