#include "../include/vec_mat.h"

void vec_mat_mul(Vec dst, Vec v, Mat m)
{
    if (m.r != dst.c) PANIC("vec_mat_mul(): destination size differs from matrix rows");
    if (m.c != v.c)   PANIC("vec_mat_mul(): source size differs from matrix columns");

    for (bsize r = 0; r < m.r; r++) {
        for (bsize c = 0; c < m.c; c++) {
            vec_el(dst, r) += mat_el(m, r, c) * vec_el(v, c);
        }
    }
}

Vec  mat_to_row_vec(Mat src, bsize r)
{
    if (r >= src.r) PANIC("mat_to_row_vec(): row index is out of bounds");

    Vec dst;

    dst.c = src.c;
    dst.s = 1;
    dst.e = &mat_el(src, r, 0);

    return dst;
}

Vec  mat_to_col_vec(Mat src, bsize c)
{
    if (c >= src.c) PANIC("mat_to_col_vec(): col index is out of bounds");

    Vec dst;

    dst.c = src.r;
    dst.s = src.c;
    dst.e = &mat_el(src, 0, c);

    return dst;
}