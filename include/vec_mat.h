#include "vec.h"
#include "mat.h"
#include "../config.h"

#ifndef _BRAINN_VEC_MAT_H_
#define _BRAINN_VEC_MAT_H_

void vec_mat_mul(Vec dst, Vec v, Mat m);
Vec  mat_to_row_vec(Mat src, bsize r);
Vec  mat_to_col_vec(Mat src, bsize c);

#endif // _BRAINN_VEC_MAT_H_