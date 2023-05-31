#include "../include/nn.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

NN nn_alloc(size_t *arch, size_t layers)
{
    assert(layers > 1 && "Minimum layer-count is 2 (inputs-ouputs)");

    NN nn;

    nn.l = layers - 1;  // don't count input layer
    nn.a = (Vec *) malloc(sizeof (*nn.a) * layers);
    nn.b = (Vec *) malloc(sizeof (*nn.b) * nn.l);
    nn.w = (Mat *) malloc(sizeof (*nn.w) * nn.l);

    assert(nn.a != NULL && nn.b != NULL && nn.w != NULL && "Failed to allocate memory for network");

    nn_input(nn) = vec_alloc(arch[0]);   // inputs
    for (size_t i = 0; i < nn.l; i++) {
        nn.a[i+1] = vec_alloc(arch[i+1]);
        nn.b[i]   = vec_alloc(arch[i+1]);
        nn.w[i]   = mat_alloc(arch[i], arch[i+1]);
    }

    return nn;
}

void nn_init(NN nn, float min, float max)
{
    assert(nn.a != NULL && nn.b != NULL && nn.w != NULL && "No memory for layers allocated");

    vec_fill(nn_input(nn), 0);
    for (size_t i = 0; i < nn.l; i++) {
        vec_fill(nn.a[i+1], 0);
        vec_fill(nn.b[i],   0);
        mat_rand(nn.w[i], min, max);
    }
}

void nn_fill(NN nn, size_t val)
{
    assert(nn.a != NULL && nn.b != NULL && nn.w != NULL && "No memory for layers allocated");

    vec_fill(nn_input(nn), val);
    for (size_t i = 0; i < nn.l; i++) {
        vec_fill(nn.a[i+1], val);
        vec_fill(nn.b[i],   val);
        mat_fill(nn.w[i],   val);
    }
}

void nn_free(NN nn)
{
    vec_free(nn_input(nn));
    for (size_t i = 0; i < nn.l; i++) {
        vec_free(nn.a[i+1]);
        vec_free(nn.b[i]);
        mat_free(nn.w[i]);
    }

    free(nn.a);
    free(nn.b);
    free(nn.w);
}


void nn_print_intern(NN nn, const char *name)
{
    assert(nn.a != NULL && nn.b != NULL && nn.w != NULL && "No memory for layers allocated");

    char buf[32];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.l; i++) {
        snprintf(buf, sizeof (buf), "w[%zu]", i);
        mat_print_intern(nn.w[i], buf, 4);
        snprintf(buf, sizeof (buf), "b[%zu]", i);
        vec_print_intern(nn.b[i], buf, 4);
    }
}
