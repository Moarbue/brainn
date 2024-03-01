#include "../include/optimizer.h"
#include <math.h>

bfloat SGD_update_bias(SGD *sgd, bfloat b, bfloat gb)
{
    return b - sgd->lr*gb;
}

bfloat SGD_update_weight(SGD *sgd, bfloat w, bfloat gw)
{
    return w - sgd->lr*gw;
}

bfloat decay_update_bias(Decay *dec, bfloat b, bfloat gb)
{
    dec->i++;
    dec->lr = 1 / (1 + dec->a*dec->i) * dec->lr0;
    return b - dec->lr*gb;
}

bfloat decay_update_weight(Decay *dec, bfloat w, bfloat gw)
{
    return w - dec->lr*gw;
}

bfloat momentum_update_bias(Momentum *mom, bfloat b, bfloat gb, bsize l, bsize c)
{
    vec_el(mom->db[l], c) = mom->a*vec_el(mom->db[l], c) - mom->lr*gb;
    return b + vec_el(mom->db[l], c);
}

bfloat momentum_update_weight(Momentum *mom, bfloat w, bfloat gw, bsize l, bsize r, bsize c)
{
    mat_el(mom->dw[l], r, c) = mom->a*mat_el(mom->dw[l], r, c) - mom->lr*gw;
    return w + mat_el(mom->dw[l], r, c);
}

bfloat adaGrad_update_bias(AdaGrad *ag, bfloat b, bfloat gb, bsize l, bsize c)
{
    vec_el(ag->Gb[l], c) += gb*gb;
    return b - ag->lr / (1e-8 + sqrt(vec_el(ag->Gb[l], c))) * gb;
}

bfloat adaGrad_update_weight(AdaGrad *ag, bfloat w, bfloat gw, bsize l, bsize r, bsize c)
{
    mat_el(ag->Gw[l], r, c) += gw*gw;
    return w - ag->lr / (1e-8 + sqrt(mat_el(ag->Gw[l], r, c))) * gw;
}

bfloat RMSProp_update_bias(RMSProp *rms, bfloat b, bfloat gb, bsize l, bsize c)
{
    vec_el(rms->vb[l], c) = rms->a*vec_el(rms->vb[l], c) + (1.0 - rms->a)*gb*gb;
    return b - rms->lr / (1e-8 + sqrt(vec_el(rms->vb[l], c))) * gb;
}

bfloat RMSProp_update_weight(RMSProp *rms, bfloat w, bfloat gw, bsize l, bsize r, bsize c)
{
    mat_el(rms->vw[l], r, c) = rms->a*mat_el(rms->vw[l], r, c) + (1.0 - rms->a)*gw*gw;
    return w - rms->lr / (1e-8 + sqrt(mat_el(rms->vw[l], r, c))) * gw;
}

bfloat adam_update_bias(Adam *ad, bfloat b, bfloat gb, bsize l, bsize c)
{
    ad->i++;
    vec_el(ad->mb[l], c) = ad->b1*vec_el(ad->mb[l], c) + (1.0 - ad->b1)*gb;
    vec_el(ad->vb[l], c) = ad->b2*vec_el(ad->vb[l], c) + (1.0 - ad->b2)*gb*gb;

    bfloat mb = vec_el(ad->mb[l], c) / (1 - pow(ad->b1, ad->i));
    bfloat vb = vec_el(ad->vb[l], c) / (1 - pow(ad->b2, ad->i));

    return b - ad->lr*mb / (1e-8 + sqrt(vb));
}

bfloat adam_update_weight(Adam *ad, bfloat w, bfloat gw, bsize l, bsize r, bsize c)
{
    mat_el(ad->mw[l], r, c) = ad->b1*mat_el(ad->mw[l], r, c) + (1.0 - ad->b1)*gw;
    mat_el(ad->vw[l], r, c) = ad->b2*mat_el(ad->vw[l], r, c) + (1.0 - ad->b2)*gw*gw;

    bfloat mw = mat_el(ad->mw[l], r, c) / (1 - pow(ad->b1, ad->i));
    bfloat vw = mat_el(ad->vw[l], r, c) / (1 - pow(ad->b2, ad->i));

    return w - ad->lr*mw / (1e-8 + sqrt(vw));
}

Optimizer optimizer_SGD(bfloat lr)
{
    Optimizer o;
    o.t = OPTIMIZER_SGD;

    o.sgd.lr = lr;

    return o;
}

Optimizer optimizer_decay(bfloat lr, bfloat a)
{
    Optimizer o;
    o.t = OPTIMIZER_DECAY;

    o.dec.lr0 = lr;
    o.dec.a   = a;
    o.dec.i   = 0;
    o.dec.lr  = 0;

    return o;
}

Optimizer optimizer_momentum(bfloat lr, bfloat a, Vec *b, Mat *w, bsize l)
{
    Optimizer o;
    o.t = OPTIMIZER_MOMENTUM;

    o.mom.lr = lr;
    o.mom.a  = a;
    o.mom.db = (Vec *) BALLOC(l * sizeof (Vec));
    o.mom.dw = (Mat *) BALLOC(l * sizeof (Mat));

    if (o.mom.db == NULL || o.mom.dw == NULL) PANIC("optimizer_momentum(): Failed to allocate memory!");

    for (bsize i = 0; i < l; i++) {
        o.mom.db[i] = vec_alloc(b[i].c);
        o.mom.dw[i] = mat_alloc(w[i].r, w[i].c);

        vec_fill(o.mom.db[i], 0);
        mat_fill(o.mom.dw[i], 0);
    }

    return o;
}

Optimizer optimizer_adaGrad(bfloat lr, Vec *b, Mat *w, bsize l)
{
    Optimizer o;
    o.t = OPTIMIZER_ADAGRAD;

    o.ag.lr = lr;
    o.ag.Gb = (Vec *) BALLOC(l * sizeof (Vec));
    o.ag.Gw = (Mat *) BALLOC(l * sizeof (Mat));

    if (o.ag.Gb == NULL || o.ag.Gw == NULL) PANIC("optimizer_adaGrad(): Failed to allocate memory!");

    for (bsize i = 0; i < l; i++) {
        o.ag.Gb[i] = vec_alloc(b[i].c);
        o.ag.Gw[i] = mat_alloc(w[i].r, w[i].c);

        vec_fill(o.ag.Gb[i], 0);
        mat_fill(o.ag.Gw[i], 0);
    }

    return o;
}

Optimizer optimizer_RMSProp(bfloat lr, bfloat a, Vec *b, Mat *w, bsize l)
{
    Optimizer o;
    o.t = OPTIMIZER_RMSPROP;

    o.rms.lr = lr;
    o.rms.a  = a;
    o.rms.vb = (Vec *) BALLOC(l * sizeof (Vec));
    o.rms.vw = (Mat *) BALLOC(l * sizeof (Mat));

    if (o.rms.vb == NULL || o.rms.vw == NULL) PANIC("optimizer_RMSProp(): Failed to allocate memory!");

    for (bsize i = 0; i < l; i++) {
        o.rms.vb[i] = vec_alloc(b[i].c);
        o.rms.vw[i] = mat_alloc(w[i].r, w[i].c);

        vec_fill(o.rms.vb[i], 0);
        mat_fill(o.rms.vw[i], 0);
    }

    return o;
}

Optimizer optimizer_adam(bfloat lr, bfloat b1, bfloat b2, Vec *b, Mat *w, bsize l)
{
    Optimizer o;
    o.t = OPTIMIZER_ADAM;

    o.ad.lr = lr;
    o.ad.b1 = b1;
    o.ad.b2 = b2;
    o.ad.i  = 0;
    o.ad.mb = (Vec *) BALLOC(l * sizeof (Vec));
    o.ad.vb = (Vec *) BALLOC(l * sizeof (Vec));
    o.ad.mw = (Mat *) BALLOC(l * sizeof (Mat));
    o.ad.vw = (Mat *) BALLOC(l * sizeof (Mat));

    if (o.ad.mb == NULL || o.ad.vb == NULL || o.ad.mw == NULL || o.ad.vw == NULL) PANIC("optimizer_adam(): Failed to allocate memory!");

    for (bsize i = 0; i < l; i++) {
        o.ad.mb[i] = vec_alloc(b[i].c);
        o.ad.vb[i] = vec_alloc(b[i].c);
        o.ad.mw[i] = mat_alloc(w[i].r, w[i].c);
        o.ad.vw[i] = mat_alloc(w[i].r, w[i].c);

        vec_fill(o.ad.mb[i], 0);
        vec_fill(o.ad.vb[i], 0);
        mat_fill(o.ad.mw[i], 0);
        mat_fill(o.ad.vw[i], 0);
    }

    return o;
}

void optimizer_free(Optimizer *o, bsize l)
{
    switch (o->t) {
        case OPTIMIZER_SGD:
        case OPTIMIZER_DECAY:
        break;

        case OPTIMIZER_MOMENTUM:
        {
            for (bsize i = 0; i < l; i++) {
                vec_free(o->mom.db[i]);
                mat_free(o->mom.dw[i]);
            }
            BFREE(o->mom.db);
            BFREE(o->mom.dw);
        }
        break;

        case OPTIMIZER_ADAGRAD:
        {
            for (bsize i = 0; i < l; i++) {
                vec_free(o->ag.Gb[i]);
                mat_free(o->ag.Gw[i]);
            }
            BFREE(o->ag.Gb);
            BFREE(o->ag.Gw);
        }
        break;

        case OPTIMIZER_RMSPROP:
        {
            for (bsize i = 0; i < l; i++) {
                vec_free(o->rms.vb[i]);
                mat_free(o->rms.vw[i]);
            }
            BFREE(o->rms.vb);
            BFREE(o->rms.vw);
        }
        break;

        case OPTIMIZER_ADAM:
        {
            for (bsize i = 0; i < l; i++) {
                vec_free(o->ad.mb[i]);
                vec_free(o->ad.vb[i]);
                mat_free(o->ad.mw[i]);
                mat_free(o->ad.vw[i]);
            }
            BFREE(o->ad.mb);
            BFREE(o->ad.vb);
            BFREE(o->ad.mw);
            BFREE(o->ad.vw);
        }
        break;
    }
}

bfloat optimizer_update_bias(Optimizer *o, bfloat b, bfloat gb, bsize l, bsize c)
{
    switch (o->t) {
        case OPTIMIZER_SGD:
        return SGD_update_bias(&o->sgd, b, gb);

        case OPTIMIZER_DECAY:
        return decay_update_bias(&o->dec, b, gb);

        case OPTIMIZER_MOMENTUM:
        return momentum_update_bias(&o->mom, b, gb, l, c);

        case OPTIMIZER_ADAGRAD:
        return adaGrad_update_bias(&o->ag, b, gb, l, c);

        case OPTIMIZER_RMSPROP:
        return RMSProp_update_bias(&o->rms, b, gb, l, c);

        case OPTIMIZER_ADAM:
        return adam_update_bias(&o->ad, b, gb, l, c);
    }

    return 0;
}

bfloat optimizer_update_weight(Optimizer *o, bfloat w, bfloat gw, bsize l, bsize r, bsize c)
{
    switch (o->t) {
        case OPTIMIZER_SGD:
        return SGD_update_weight(&o->sgd, w, gw);

        case OPTIMIZER_DECAY:
        return decay_update_weight(&o->dec, w, gw);

        case OPTIMIZER_MOMENTUM:
        return momentum_update_weight(&o->mom, w, gw, l, r, c);

        case OPTIMIZER_ADAGRAD:
        return adaGrad_update_weight(&o->ag, w, gw, l, r, c);

        case OPTIMIZER_RMSPROP:
        return RMSProp_update_weight(&o->rms, w, gw, l, r, c);

        case OPTIMIZER_ADAM:
        return adam_update_weight(&o->ad, w, gw, l, r, c);
    }

    return 0;
}
