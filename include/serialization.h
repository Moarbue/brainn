#include "buffer.h"
#include "vec.h"
#include "mat.h"
#include "nn.h"
#include "../config.h"

#ifndef _BRAINN_SERIALIZATION_H_
#define _BRAINN_SERIALIZATION_H_

void serialize_string(Buffer *b, const char *str, bsize length);
void serialize_bsize(Buffer *b, bsize v);
void serialize_bfloat(Buffer *b, bfloat v);
void serialize_vec(Buffer *b, Vec v);
void serialize_mat(Buffer *b, Mat m);
void serialize_nn(Buffer *b, NN nn);

char *deserialize_string(Buffer *b, bsize length);
bsize deserialize_bsize(Buffer *b);
bfloat deserialize_bfloat(Buffer *b);
void deserialize_vec(Buffer *b, Vec *v);
void deserialize_mat(Buffer *b, Mat *m);
NN deserialize_nn(Buffer *b);

#endif // _BRAINN_SERIALIZATION_H_