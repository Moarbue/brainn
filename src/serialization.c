#include "../include/serialization.h"
#include <string.h>

void serialize_string(Buffer *b, const char *str, bsize length)
{
    buffer_add(b, (uint8_t *) str, strnlen(str, length));
}

void serialize_bsize(Buffer *b, bsize v)
{
    union {
        bsize v;
        uint8_t b[sizeof (bsize)];
    } u = {.v = v};
    
    buffer_add(b, u.b, sizeof (bsize));
}

void serialize_bfloat(Buffer *b, bfloat v)
{
    union {
        bfloat v;
        uint8_t b[sizeof (bfloat)];
    } u = {.v = v};
    
    buffer_add(b, u.b, sizeof (bfloat));
}

void serialize_vec(Buffer *b, Vec v)
{
    serialize_bsize(b, v.c);
    for (bsize i = 0; i < v.c; i++) {
        serialize_bfloat(b, vec_el(v, i));
    }
}

void serialize_mat(Buffer *b, Mat m)
{
    serialize_bsize(b, m.r);
    serialize_bsize(b, m.c);
    for (bsize r = 0; r < m.r; r++) {
        for (bsize c = 0; c < m.c; c++) {
            serialize_bfloat(b, mat_el(m, r, c));
        }
    }
}

void serialize_nn(Buffer *b, NN nn)
{
    bsize *arch, layers;
    nn_get_arch(nn, &arch, &layers);
    
    serialize_bsize(b, layers);
    for (bsize l = 0; l < layers; l++) serialize_bsize(b, arch[l]);

    for (bsize l = 0; l < nn.l; l++) {
        serialize_vec(b, nn.b[l]);
        serialize_mat(b, nn.w[l]);
    }

    free(arch);
}


char *deserialize_string(Buffer *b, bsize length)
{
    return (char *) buffer_advance(b, length);
}

bsize deserialize_bsize(Buffer *b)
{
    union {
        bsize v;
        uint8_t b[sizeof (bsize)];
    } u;
    memcpy(u.b, buffer_advance(b, sizeof (bsize)), sizeof (bsize));

    return u.v;
}

bfloat deserialize_bfloat(Buffer *b)
{
    union {
        bfloat v;
        uint8_t b[sizeof (bfloat)];
    } u;
    memcpy(u.b, buffer_advance(b, sizeof (bfloat)), sizeof (bfloat));

    return u.v;
}

void deserialize_vec(Buffer *b, Vec *v)
{
    bsize c = deserialize_bsize(b);
    if (c != v->c) PANIC("deserialize_vec(): dst_vec size differs from deserialized vec %zu != %zu", v->c, c);

    for (bsize i = 0; i < v->c; i++) {
        vec_el(*v, i) = deserialize_bfloat(b);
    }
}

void deserialize_mat(Buffer *b, Mat *m)
{
    bsize r = deserialize_bsize(b);
    bsize c = deserialize_bsize(b);
    if (r != m->r) PANIC("deserialize_mat(): dst_mat rows differ from deserialized mat %zu != %zu", m->r, r);
    if (c != m->c) PANIC("deserialize_mat(): dst_mat cols differ from deserialized mat %zu != %zu", m->c, c);

    for (bsize r = 0; r < m->r; r++) {
        for (bsize c = 0; c < m->c; c++) {
            mat_el(*m, r, c) = deserialize_bfloat(b);
        }
    }
}

NN deserialize_nn(Buffer *b)
{
    bsize *arch, layers;
    layers = deserialize_bsize(b);
    arch   = (bsize *) BALLOC(layers * sizeof (bsize));
    for (bsize l = 0; l < layers; l++) arch[l] = deserialize_bsize(b);

    NN nn;
    nn = nn_alloc(arch, layers);
    nn_init(nn, -1, 1);

    for (bsize l = 0; l < nn.l; l++) {
        deserialize_vec(b, &nn.b[l]);
        deserialize_mat(b, &nn.w[l]);
    }

    free(arch);

    return nn;
}