#include "../include/activation.h"
#include <math.h>

bfloat Sigmoid(bfloat z)
{
    return 1.f / (1.f + expf(-z));
}

bfloat Tanh(bfloat z)
{
    return tanhf(z);
}

bfloat ReLU(bfloat z)
{
    return fmaxf(0, z);
}

bfloat Heaviside(bfloat z)
{
    return (z > 0);
}

bfloat GELU(bfloat z)
{
    static const bfloat SQRT2 = 1.41421356237f;

    return 0.5f * z * (1 + erff(z / SQRT2));
}

bfloat Softplus(bfloat z)
{
    return logf(1 + expf(z));
}

bfloat lReLU(bfloat z)
{
    if (z < 0) return 0.01f * z;

    return z;
}


bfloat dSigmoid(bfloat z)
{
    bfloat a = Sigmoid(z);

    return a * (1 - a);
}

bfloat dTanh(bfloat z)
{
    bfloat a = Tanh(z);

    return 1 - a*a;
}

bfloat dReLU(bfloat z)
{
    return (z > 0);
}

bfloat dHeaviside(bfloat z)
{
    (void) z;
    return 0;
}

bfloat dGELU(bfloat z)
{
    static const bfloat SQRTPI = 1.77245385091f;

    return 2.f / SQRTPI * expf(-z*z);
}

bfloat dSoftplus(bfloat z)
{
    return Sigmoid(z);
}

bfloat dlReLU(bfloat z)
{
    if (z < 0) return 0.01f;

    return 1;
}


void vec_activate(Vec v, activation_function *af)
{
    for (bsize i = 0; i < v.s; i++) {
        vec_el(v, i) = (*af)(vec_el(v, i));
    }
}