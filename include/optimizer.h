#include "../config.h"
#include "vec.h"
#include "mat.h"

#ifndef _BRAINN_OPTIMIZER_H_
#define _BRAINN_OPTIMIZER_H_

enum Optimizer_Type {
    OPTIMIZER_SGD,
    OPTIMIZER_DECAY,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_ADAGRAD,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM,
};

typedef struct {
    bfloat lr;  // learning rate
} SGD;

bfloat SGD_update_bias(SGD *sgd, bfloat b, bfloat gb);
bfloat SGD_update_weight(SGD *sgd, bfloat w, bfloat gw);


typedef struct {
    bfloat lr0; // initial learning rate
    bfloat lr;  // learning rate
    bfloat a;   // decay factor
    bsize  i;   // current epoch
} Decay;

bfloat decay_update_bias(Decay *dec, bfloat b, bfloat gb);
bfloat decay_update_weight(Decay *dec, bfloat w, bfloat gw);


typedef struct {
    bfloat lr;  // learning rate
    bfloat a;   // decay factor
    Vec *db;    // delta biases
    Mat *dw;    // delta weights
} Momentum;

bfloat momentum_update_bias(Momentum *mom, bfloat b, bfloat gb, bsize l, bsize c);
bfloat momentum_update_weight(Momentum *mom, bfloat w, bfloat gw, bsize l, bsize r, bsize c);


typedef struct {
    bfloat lr;  // learning rate
    Vec *Gb;    // sum of gradient squares for biases
    Mat *Gw;    // sum of gradient squares for weights
} AdaGrad;

bfloat adaGrad_update_bias(AdaGrad *ag, bfloat b, bfloat gb, bsize l, bsize c);
bfloat adaGrad_update_weight(AdaGrad *ag, bfloat w, bfloat gw, bsize l, bsize r, bsize c);


typedef struct {
    bfloat lr;  // learning rate
    bfloat a;   // forgetting factor
    Vec *vb;    // average of magnitudes of recent gradients for biases
    Mat *vw;    // average of magnitudes of recent gradients for weights
} RMSProp;

bfloat RMSProp_update_bias(RMSProp *rms, bfloat b, bfloat gb, bsize l, bsize c);
bfloat RMSProp_update_weight(RMSProp *rms, bfloat w, bfloat gw, bsize l, bsize r, bsize c);


typedef struct {
    bfloat lr;      // learning rate
    bfloat i;       // current epoch
    bfloat b1, b2;  // exponential decay rates for moment estimates
    Vec *mb, *vb;   // first and second momentum vectors for biases
    Mat *mw, *vw;   // first and second momentum vectors for weights
} Adam;

bfloat adam_update_bias(Adam *ad, bfloat b, bfloat gb, bsize l, bsize c);
bfloat adam_update_weight(Adam *ad, bfloat w, bfloat gw, bsize l, bsize r, bsize c);


typedef struct {
    enum Optimizer_Type t;
    union {
        SGD sgd;
        Decay dec;
        Momentum mom;
        AdaGrad ag;
        RMSProp rms;
        Adam ad;
    };
} Optimizer;

Optimizer optimizer_SGD(bfloat lr);
Optimizer optimizer_decay(bfloat lr, bfloat a);
Optimizer optimizer_momentum(bfloat lr, bfloat a, Vec *b, Mat *w, bsize l);
Optimizer optimizer_adaGrad(bfloat lr, Vec *b, Mat *w, bsize l);
Optimizer optimizer_RMSProp(bfloat lr, bfloat a, Vec *b, Mat *w, bsize l);
Optimizer optimizer_adam(bfloat lr, bfloat b1, bfloat b2, Vec *b, Mat *w, bsize l);
void      optimizer_free(Optimizer *o, bsize l);

bfloat    optimizer_update_bias(Optimizer *o, bfloat b, bfloat gb, bsize l, bsize c);
bfloat    optimizer_update_weight(Optimizer *o, bfloat w, bfloat gw, bsize l, bsize r, bsize c);

#endif // _BRAINN_OPTIMIZER_H_