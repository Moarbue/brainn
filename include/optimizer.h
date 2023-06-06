#include "vec.h"
#include "mat.h"

#ifndef BRAINN_OPTIMIZER_H_
#define BRAINN_OPTIMIZER_H_

typedef enum {
    OPTIMIZER_SDG = 0,
    OPTIMIZER_SDG_WITH_DECAY,
} Optimizer_Types;

typedef struct {
    float lr;   // learning rate
} SGD;  // Stochastic Gradient Descent

typedef struct {
    float lr;   // learning rate
    float i;    // iterations
    float d;    // learning rate decay
} SGDD; // Stochastic Gradient Descent with Learning Rate Decay

typedef struct {
    Optimizer_Types t;
    union {
        SGD  a;
        SGDD b;
    } o;
} Optimizer;

Optimizer optimizer_sdg_init (float learning_rate);
Optimizer optimizer_sdgd_init(float learning_rate, float decay);
float optimizer_update_param(Optimizer *o, size_t current_layer, size_t current_neuron, size_t previous_neuron);

#endif // BRAINN_OPTIMIZER_H_