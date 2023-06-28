#include "vec.h"

#ifndef BRAINN_ACTIVATION_H_
#define BRAINN_ACTIVATION_H_

typedef float (activation_function)(float);
typedef float (hidden_activation_function) (float);
typedef float (dhidden_activation_function)(float);
typedef float (output_activation_function) (float);
typedef float (doutput_activation_function)(float);

float Sigmoid(float x);
float Tanh(float x);
float ReLU(float x);
float Heaviside(float x);
float GELU(float x);
float Softplus(float x);
float lReLU(float x);

float dSigmoid(float x);
float dTanh(float x);
float dReLU(float x);
float dHeaviside(float x);
float dGELU(float x);
float dSoftplus(float x);
float dlReLU(float x);

void vec_activate(Vec v, activation_function *af);

#endif // BRAINN_ACTIVATION_H_