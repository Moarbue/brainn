#include <stddef.h>

#ifndef BRAINN_MAT_H_
#define BRAINN_MAT_H_

#define mat_el(m, row, col) (m).e[(row) * ((m).c + (m).s) + (col)]
#define mat_print(m, pad) mat_print_intern(m, #m, pad)
typedef struct {
    size_t r;   // rows of matrix
    size_t c;   // columns of matrix
    size_t s;   // row-stride
    float *e;   // elements
} Mat;

Mat  mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float val);
void mat_rand(Mat m, float min, float max);
Mat  mat_sub_mat(Mat m, size_t rows, size_t cols, size_t row_offset, size_t col_offset);
void mat_free(Mat m);

void mat_print_intern(Mat m, const char *name, size_t pad);

#endif // BRAINN_MAT_H_
