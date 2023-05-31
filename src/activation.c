#include "../include/activation.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>

void sigmoid(Vec v)
{
    assert(v.e != NULL && "No memory for elements allocated!");

    for (size_t i = 0; i < v.c; i++) {
        vec_el(v, i) = 1.f / (1.f + expf(-vec_el(v, i)));
    }
}

void ReLU(Vec v)
{
    assert(v.e != NULL && "No memory for elements allocated!");

    for (size_t i = 0; i < v.c; i++) {
        vec_el(v, i) = fmax(0, vec_el(v, i));
    }
}

void softmax(Vec v)
{
    assert(v.e != NULL && "No memory for elements allocated!");

    float sum = 0.f;
    float max = vec_max(v);
    for (size_t i = 0; i < v.c; i++) {
        vec_el(v, i) = expf(vec_el(v, i) - max);
        sum += vec_el(v, i);
    }

    for (size_t i = 0; i < v.c; i++) {
        vec_el(v, i) /= sum;
    }
}
