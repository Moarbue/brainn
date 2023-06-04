#include "../include/activation.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

float Sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

float Tanh(float x)
{
    return tanhf(x);
}

float ReLU(float x)
{
    return fmaxf(0, x);
}

float Heaviside(float x)
{
    return (x > 0);
}

float GELU(float x)
{
    static const float SQRT2 = 1.41421356237f;

    return 0.5f * x * (1 + erff(x / SQRT2));
}

float Softplus(float x)
{
    return logf(1 + expf(x));
}

float lRELU(float x)
{
    if (x < 0) return 0.01f * x;

    return x;
}


float dSigmoid(float x)
{
    float y = Sigmoid(x);

    return y * (1 - y);
}

float dTanh(float x)
{
    float y = Tanh(x);

    return 1 - y*y;
}

float dReLU(float x)
{
    return (x > 0);
}

float dHeaviside(float x)
{
    (void) x;
    return 0;
}

float dGELU(float x)
{
    static const float SQRTPI = 1.77245385091f;

    return 2.f / SQRTPI * expf(-x*x);
}

float dSoftplus(float x)
{
    return Sigmoid(x);
}

float dlRELU(float x)
{
    if (x < 0) return 0.01f;

    return 1;
}


void vec_activate(Vec v, activation_function *af)
{
    for (size_t i = 0; i < v.c; i++) {
        vec_el(v, i) = (*af)(vec_el(v, i));
    }
}
