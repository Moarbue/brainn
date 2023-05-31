#include <stddef.h>

#ifndef BRAINN_MAT_H_
#define BRAINN_MAT_H_

#define mat_el(m, row, col) (m).e[(row) * (m).c + (col)]
typedef struct {
    size_t r;   // rows of matrix
    size_t c;   // columns of matrix
    float *e;   // elements
} Mat;

Mat  mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float val);
void mat_rand(Mat m, float min, float max);
void mat_free(Mat m);

#endif // BRAINN_MAT_H_
