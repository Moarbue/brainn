#include "../config.h"
#include "vec.h"

#ifndef _BRAINN_ACTIVATION_H_
#define _BRAINN_ACTIVATION_H_

typedef void (hidden_activation) (Vec z);
typedef void (dhidden_activation)(Vec z, Vec a);
typedef void (output_activation) (Vec z);
typedef void (doutput_activation)(Vec z, Vec a);

void Sigmoid(Vec z);
void Tanh(Vec z);
void ReLU(Vec z);
void Heaviside(Vec z);
void GELU(Vec z);
void Softplus(Vec z);
void lReLU(Vec z);
void Softmax(Vec z);

void dSigmoid(Vec z, Vec a);
void dTanh(Vec z, Vec a);
void dReLU(Vec z, Vec a);
void dHeaviside(Vec z, Vec a);
void dGELU(Vec z, Vec a);
void dSoftplus(Vec z, Vec a);
void dlReLU(Vec z, Vec a);
void dSoftmax(Vec z, Vec a);

#endif // _BRAINN_ACTIVATION_H_