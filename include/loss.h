#ifndef BRAINN_LOSS_H_
#define BRAINN_LOSS_H_

typedef float (loss_function)(float, float);

float SEL(float t, float x);
float CEL(float t, float x);

#endif // BRAINN_LOSS_H_