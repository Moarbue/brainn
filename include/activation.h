#include "vec.h"

#ifndef BRAINN_ACTIVATION_H_
#define BRAINN_ACTIVATION_H_

typedef void (hidden_activation_function)(Vec);
typedef void (output_activation_function)(Vec);

void sigmoid(Vec v);
void ReLU(Vec v);
void softmax(Vec v);

#endif // BRAINN_ACTIVATION_H_