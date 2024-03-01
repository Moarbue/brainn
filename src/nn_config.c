#include "../include/nn.h"

void nn_set_activation_function(NN *nn, hidden_activation hf, dhidden_activation dhf, 
                                        output_activation of, doutput_activation dof)
{
    nn->hf  = hf;
    nn->dhf = dhf;
    nn->of  = of;
    nn->dof = dof;
}

void nn_set_loss_function(NN *nn, loss_function C, dloss_function dC)
{
    nn->C  = C;
    nn->dC = dC;
}

void nn_get_arch(NN nn, bsize *arch[], bsize *layers)
{
    bsize *a, L;

    L = nn.l + 1;
    a = BALLOC(L * sizeof (bsize));

    for (bsize l = 0; l < L; l++) {
        a[l] = nn.a[l].c;
    }

    *arch   = a;
    *layers = L;
}

void nn_set_optimizer(NN *nn, Optimizer o)
{
    *nn->o = o;
}
