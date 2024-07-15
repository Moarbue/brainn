#include "../include/activation.h"
#include <math.h>

void Sigmoid(Vec z)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = 1.0 / (1.0 + exp(-vec_el(z, i)));
    }
}

void Tanh(Vec z)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = tanh(vec_el(z, i));
    }
}

void ReLU(Vec z)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = fmax(0, vec_el(z, i));
    }
}

void Heaviside(Vec z)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = (vec_el(z, i) > 0);
    }
}

void GELU(Vec z)
{
    static const bfloat SQRT2 = 1.41421356237;

    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = 0.5 * vec_el(z, i) * (1.0 + erf(vec_el(z, i) / SQRT2));
    }
}

void Softplus(Vec z)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = log(1 + exp(vec_el(z, i)));
    }
}

void lReLU(Vec z)
{
    for (bsize i = 0; i < z.c; i++) {
        if (vec_el(z, i) < 0)
            vec_el(z, i) = 0.01 * vec_el(z, i);
    }
}

void Softmax(Vec z)
{
    bfloat D = -vec_max(z);

    bfloat s = 0.0;
    for (bsize i = 0; i < z.c; i++) {
        s += exp(vec_el(z, i) + D);
    }

    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = exp(vec_el(z, i) + D) / s;
    }
}


void dSigmoid(Vec z, Vec a)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = vec_el(a, i) * (1 - vec_el(a, i));
    }
}

void dTanh(Vec z, Vec a)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = 1 - vec_el(a, i)*vec_el(a, i);
    }
}

void dReLU(Vec z, Vec a)
{
    (void) a;
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = (vec_el(z, i) > 0);
    }
}

void dHeaviside(Vec z, Vec a)
{
    (void) a;
    vec_fill(z, 0);
}

void dGELU(Vec z, Vec a)
{
    static const bfloat SQRTPI = 1.77245385091;

    (void) a;

    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = 2.0 / SQRTPI * exp(-vec_el(z, i) * vec_el(z, i));
    }
}

void dSoftplus(Vec z, Vec a)
{
    (void) a;
    Sigmoid(z);
}

void dlReLU(Vec z, Vec a)
{
    (void) a;

    for (bsize i = 0; i < z.c; i++) {
        if (vec_el(z, i) < 0)
            vec_el(z, i) = 0.01;
        else
            vec_el(z, i) = 1.0;
    }
}

void dSoftmax(Vec z, Vec a)
{
    for (bsize i = 0; i < z.c; i++) {
        vec_el(z, i) = 0;
        for (bsize j = 0; j < z.c; j++) {
            vec_el(z, i) += vec_el(a, j) * ((i == j) - vec_el(a, i));
        }
    }
}