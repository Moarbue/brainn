#include "../config.h"

#ifndef _BRAINN_MAT_H_
#define _BRAINN_MAT_H_

#define mat_el(m, row, col) (m).e[(row) * (m).c + (col)]
#define mat_print(m, pad) mat_print_intern(m, #m, pad)

typedef struct {
    bsize r;    // rows
    bsize c;    // columns
    bfloat *e;  // elements
} Mat;

Mat  mat_alloc(bsize r, bsize c);
void mat_fill(Mat m, bfloat value);
void mat_rand(Mat m, bfloat min, bfloat max);
void mat_free(Mat m);

void mat_print_intern(Mat m, const char *name, bsize pad);

#endif // _BRAINN_MAT_H_