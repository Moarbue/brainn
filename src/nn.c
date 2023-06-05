#include "../include/nn.h"
#include "../include/vec_mat.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

NN nn_alloc(size_t *arch, size_t layers)
{
    assert(layers > 1 && "Minimum layer-count is 2 (inputs-ouputs)");

    NN nn;

    nn.l  = layers - 1;  // don't count input layer
    nn.a  = (Vec *) malloc(sizeof (*nn.a)  * layers);
    nn.b  = (Vec *) malloc(sizeof (*nn.b)  * nn.l);
    nn.w  = (Mat *) malloc(sizeof (*nn.w)  * nn.l);
    nn.da = (Vec *) malloc(sizeof (*nn.da) * nn.l);
    nn.ga = (Vec *) malloc(sizeof (*nn.ga) * layers);
    nn.gb = (Vec *) malloc(sizeof (*nn.gb) * nn.l);
    nn.gw = (Mat *) malloc(sizeof (*nn.gw) * nn.l);
    nn.gc = (size_t *) malloc(sizeof(*nn.gc));

    assert(nn.a  != NULL && nn.b  != NULL && nn.w  != NULL && "Failed to allocate memory for network");
    assert(nn.da != NULL && "Failed to allocate memory for network");
    assert(nn.ga != NULL && nn.gb != NULL && nn.gw != NULL && "Failed to allocate memory for network");

    nn.a [0] = vec_alloc(arch[0]);
    nn.ga[0] = vec_alloc(arch[0]);
    for (size_t i = 0; i < nn.l; i++) {
        nn.a[i+1]  = vec_alloc(arch[i+1]);
        nn.b[i]    = vec_alloc(arch[i+1]);
        nn.w[i]    = mat_alloc(arch[i], arch[i+1]);
        nn.da[i]   = vec_alloc(arch[i+1]);
        nn.ga[i+1] = vec_alloc(arch[i+1]);
        nn.gb[i]   = vec_alloc(arch[i+1]);
        nn.gw[i]   = mat_alloc(arch[i], arch[i+1]);
    }
    *nn.gc = 0;

    nn.haf  = ReLU;
    nn.dhaf = dReLU;
    nn.oaf  = Sigmoid;
    nn.doaf = dSigmoid;
    nn.lf   = CEL;
    nn.dlf  = dCEL;

    return nn;
}

void nn_init(NN nn, float min, float max)
{
    assert(nn.a  != NULL && nn.b  != NULL && nn.w  != NULL && "Failed to allocate memory for network");
    assert(nn.da != NULL && "Failed to allocate memory for network");
    assert(nn.ga != NULL && nn.gb != NULL && nn.gw != NULL && "Failed to allocate memory for network");

    vec_fill(nn.a [0], 0);
    vec_fill(nn.ga[0], 0);
    for (size_t i = 0; i < nn.l; i++) {
        vec_fill(nn.a [i+1], 0);
        vec_fill(nn.b [i],   0);
        mat_rand(nn.w [i], min, max);
        vec_fill(nn.da[i],   0);
        vec_fill(nn.ga[i+1], 0);
        vec_fill(nn.gb[i],   0);
        mat_fill(nn.gw[i],   0);
    }
}

void nn_fill(NN nn, size_t val)
{
    assert(nn.a  != NULL && nn.b  != NULL && nn.w  != NULL && "Failed to allocate memory for network");
    assert(nn.da != NULL && "Failed to allocate memory for network");
    assert(nn.ga != NULL && nn.gb != NULL && nn.gw != NULL && "Failed to allocate memory for network");

    vec_fill(nn.a [0], 0);
    vec_fill(nn.ga[0], 0);
    for (size_t i = 0; i < nn.l; i++) {
        vec_fill(nn.a [i+1], val);
        vec_fill(nn.b [i],   val);
        mat_fill(nn.w [i],   val);
        vec_fill(nn.da[i],   val);
        vec_fill(nn.ga[i+1], val);
        vec_fill(nn.gb[i],   val);
        mat_fill(nn.gw[i],   val);
    }
}

Vec nn_forward(NN nn, Vec input)
{
    assert(nn.a  != NULL && nn.b  != NULL && nn.w  != NULL && "Failed to allocate memory for network");
    assert(nn.da != NULL && "Failed to allocate memory for network");
    assert(nn.ga != NULL && nn.gb != NULL && nn.gw != NULL && "Failed to allocate memory for network");

    vec_copy(nn_input(nn), input);
    for (size_t i = 0; i < nn.l; i++) {
        vec_mat_mul(nn.a[i+1], nn.a[i], nn.w[i]);
        vec_sum(nn.a[i+1], nn.b[i]);
        vec_copy(nn.da[i], nn.a[i+1]);
        
        if (i < (nn.l - 1))
            vec_activate(nn.a[i+1], nn.haf);
        else
            vec_activate(nn.a[i+1], nn.oaf);
    }
    
    return nn_output(nn);
}

void nn_backpropagate(NN nn, Vec output)
{
    assert(nn.a  != NULL && nn.b  != NULL && nn.w  != NULL && "Failed to allocate memory for network");
    assert(nn.da != NULL && "Failed to allocate memory for network");
    assert(nn.ga != NULL && nn.gb != NULL && nn.gw != NULL && "Failed to allocate memory for network");
    assert(nn_output(nn).c == output.c && "Network output and expected output have wrong dimensions");

    size_t lc, nc, pnc;
    
    // clear previous activation gradients
    lc = nn.l;
    for (size_t l = 0; l < lc; l++) {
        vec_fill(nn.ga[l], 0);
    }

    // kickoff backpropagation by calculating the derivative of the loss function
    nc = nn_output(nn).c;
    for (size_t n = 0; n < nc; n++) {
        vec_el(nn.ga[lc], n) = (*nn.dlf)(vec_el(output, n), vec_el(nn_output(nn), n));
    }

    // traverse network backwards
    for (size_t l = lc; l > 0; l--) {
        nc = nn.a[l].c;

        // for each neuron n in current layer
        for (size_t n = 0; n < nc; n++) {
            float dact = (l == lc) ? (*nn.doaf)(vec_el(nn.da[l-1], n)) : (*nn.dhaf)(vec_el(nn.da[l-1], n));
            float daa  = vec_el(nn.ga[l], n);
            vec_el(nn.gb[l-1], n) += daa * dact;

            // for each neuron p in the previous layer
            pnc = nn.a[l-1].c;
            for (size_t p = 0; p < pnc; p++) {
                float pa = vec_el(nn.a[l-1], p);
                float  w = mat_el(nn.w[l-1], p, n);

                vec_el(nn.ga[l-1], p)    += daa * dact * w;
                mat_el(nn.gw[l-1], p, n) += daa * dact * pa;
            }
        }
    }
    (*nn.gc)++;
}

void nn_evolve(NN nn, float learning_rate)
{
    assert(nn.a  != NULL && nn.b  != NULL && nn.w  != NULL && "Failed to allocate memory for network");
    assert(nn.ga != NULL && nn.gb != NULL && nn.gw != NULL && "Failed to allocate memory for network");

    size_t lc, nc, pnc;

    lc = nn.l;
    for (size_t l = 0; l < lc; l++) {
        nc = nn.w[l].c;
        for (size_t n = 0; n < nc; n++) {
            vec_el(nn.b[l], n) -= vec_el(nn.gb[l], n) / *nn.gc * learning_rate;
            pnc = nn.w[l].r;
            for (size_t p = 0; p < pnc; p++) {
                mat_el(nn.w[l], p, n) -= mat_el(nn.gw[l], p, n) / *nn.gc * learning_rate;
            }
        }
        *nn.gc = 0;
        vec_fill(nn.gb[l], 0);
        mat_fill(nn.gw[l], 0);
    }
}

float nn_loss(NN nn, Mat training_inputs, Mat expected_outputs)
{
    assert(nn_input(nn).c    == training_inputs.c  && "Training data input matrix has wrong dimensions");
    assert(nn_output(nn).c   == expected_outputs.c && "Expected output matrix has wrong dimensions");
    assert(training_inputs.r == expected_outputs.r && "Training data input matrix and expected output matrix don't have the same dimensions");

    size_t samples = training_inputs.r;
    float cost = 0.f;
    for (size_t i = 0; i < samples; i++) {
        Vec input, output;
        input  = mat_to_row_vec(training_inputs,  i);
        output = mat_to_row_vec(expected_outputs, i);

        nn_forward(nn, input);
        for (size_t j = 0; j < nn_output(nn).c; j++) {
            cost += (*nn.lf)(vec_el(output, j), vec_el(nn_output(nn), j));
        }
    }
    cost /= (float) samples;

    return cost;
}

void nn_free(NN nn)
{
    vec_free(nn.a [0]);
    vec_free(nn.ga[0]);
    for (size_t i = 0; i < nn.l; i++) {
        vec_free(nn.a [i+1]);
        vec_free(nn.b [i]);
        mat_free(nn.w [i]);
        vec_free(nn.ga[i+1]);
        vec_free(nn.gb[i]);
        mat_free(nn.gw[i]);
    }

    free(nn.a);
    free(nn.b);
    free(nn.w);
    free(nn.ga);
    free(nn.gb);
    free(nn.gw);
    free(nn.gc);
}


void nn_set_activation_function(NN *nn, hidden_activation_function *haf, dhidden_activation_function *dhaf,
                                        output_activation_function *oaf, doutput_activation_function *doaf)
{
    nn->haf  = haf;
    nn->dhaf = dhaf;
    nn->oaf  = oaf;
    nn->doaf = doaf;
}

void nn_set_loss_functions(NN *nn, loss_function *lf, dloss_function *dlf)
{
    nn->lf = lf;
    nn->dlf = dlf;
}


void nn_print_intern(NN nn, const char *name)
{
    assert(nn.a  != NULL && nn.b  != NULL && nn.w  != NULL && "Failed to allocate memory for network");

    char buf[32];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.l; i++) {
        snprintf(buf, sizeof (buf), "w[%zu]", i);
        mat_print_intern(nn.w[i], buf, 4);
        snprintf(buf, sizeof (buf), "b[%zu]", i);
        vec_print_intern(nn.b[i], buf, 4);
    }
}
