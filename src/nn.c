#include "../include/nn.h"
#include <math.h>

float Sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

NN nn_alloc(bsize *arch, bsize layers)
{
    if (layers < 1) PANIC("nn_alloc(): Minimum layer count is 1!");

    NN nn;

    nn.l = layers - 1;
    nn.a = (Vec*) BALLOC(layers * sizeof (Vec));
    nn.b = (Vec*) BALLOC(nn.l   * sizeof (Vec));
    nn.w = (Mat*) BALLOC(nn.l   * sizeof (Mat));

    if (nn.a == NULL || nn.b == NULL || nn.w == NULL) PANIC("nn_alloc(): Failed to allocate memory!");

    nn_input(nn) = vec_alloc(arch[0]);
    arch++;

    for (bsize l = 0; l < nn.l; l++) {
        nn.a[l+1] = vec_alloc(arch[l]);
        nn.b[l]   = vec_alloc(arch[l]);
        nn.w[l]   = mat_alloc(arch[l], arch[l-1]);
    }

    return nn;
}

void nn_init(NN nn, bfloat min, bfloat max)
{
    vec_fill(nn_input(nn), 0);
    for (bsize l = 0; l < nn.l; l++) {
        vec_fill(nn.a[l+1], 0);
        vec_fill(nn.b[l],   0);
        mat_rand(nn.w[l], min, max);
    }
}

Vec nn_forward(NN nn, Vec input)
{
    vec_copy(nn_input(nn), input);
    for (bsize l = 0; l < nn.l; l++) {
        vec_mat_mul(nn.a[l+1], nn.a[l], nn.w[l]);
        vec_sum(nn.a[l+1], nn.b[l]);
        for (bsize i = 0; i < nn.a[l+1].s; i++) vec_el(nn.a[l+1], i) = Sigmoid(vec_el(nn.a[l+1], i));
    }

    return nn_output(nn);
}

void nn_free(NN nn)
{
    vec_free(nn_input(nn));
    for (bsize l = 0; l < nn.l; l++) {
        vec_free(nn.a[l+1]);
        vec_free(nn.b[l]);
        mat_free(nn.w[l]);
    }
}


void nn_print_intern(NN nn, const char *name)
{
    char buf[32];
    printf("%s = [\n", name);
    for (bsize i = 0; i < nn.l; i++) {
        snprintf(buf, sizeof (buf), "w[%zu]", i);
        mat_print_intern(nn.w[i], buf, 4);
        snprintf(buf, sizeof (buf), "b[%zu]", i);
        vec_print_intern(nn.b[i], buf, 4);
    }
    printf("]\n");
}