#include "../include/optimizer.h"

#include <stdint.h>

Optimizer optimizer_sdg_init (float learning_rate)
{
    Optimizer o;

    o.t   = OPTIMIZER_SDG;
    o.o.a = (SGD) { .lr = learning_rate };

    return o;
}

Optimizer optimizer_sdgd_init(float learning_rate, float decay)
{
    Optimizer o;

    o.t   = OPTIMIZER_SDG_WITH_DECAY;
    o.o.b = (SGDD) { .lr = learning_rate, .d = decay, .i = 0 };

    return o;
}

float optimizer_update_param(Optimizer *o, size_t current_layer, size_t current_neuron, size_t previous_neuron)
{
    (void) current_layer;
    (void) current_neuron;
    (void) previous_neuron;
    float res = 0.f;

    switch (o->t) {
        case OPTIMIZER_SDG:
        {
            res = o->o.a.lr;
        }
        break;
        case OPTIMIZER_SDG_WITH_DECAY:
        {
            res = o->o.b.lr * (1.f / (1 + o->o.b.d * o->o.b.i));
            o->o.b.i++;
        }
        break;
    }

    return res;
}
