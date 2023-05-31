#include "vec.h"

#ifndef BRAINN_ACTIVATION_H_
#define BRAINN_ACTIVATION_H_

typedef void (hidden_activation_function)(Vec);
typedef void (output_activation_function)(Vec);

void Sigmoid(Vec v);
void Tanh(Vec v);
void ReLU(Vec v);
void Softmax(Vec v);

#endif // BRAINN_ACTIVATION_H_