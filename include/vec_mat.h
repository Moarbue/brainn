#include "vec.h"
#include "mat.h"

#ifndef BRAINN_VEC_MAT_H_
#define BRAINN_VEC_MAT_H_

void vec_mat_mul(Vec dst, Vec v, Mat m);
Vec  mat_to_row_vec(Mat m, size_t row);
Vec  mat_to_col_vec(Mat m, size_t col);

#endif // BRAINN_VEC_MAT_H_