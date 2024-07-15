#include "../config.h"
#include "vec.h"
#include "mat.h"

#ifndef _BRAINN_VEC_MAT_H_
#define _BRAINN_VEC_MAT_H_

void vec_mat_mul(Vec dst, Vec v, Mat m);
Vec  mat_to_row_vec(Mat src, bsize r);
Vec  mat_to_col_vec(Mat src, bsize c);
void vec_to_mat_row(Mat dst, Vec v, bsize r);
void vec_to_mat_col(Mat dst, Vec v, bsize c);

#endif // _BRAINN_VEC_MAT_H_