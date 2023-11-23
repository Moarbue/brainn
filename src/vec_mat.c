#include "../include/vec_mat.h"

void vec_mat_mul(Vec dst, Vec v, Mat m)
{
    if (m.r != dst.s) PANIC("vec_mat_mul(): destination size differs from matrix rows");
    if (m.c != v.s)   PANIC("vec_mat_mul(): source size differs from matrix columns");

    for (bsize r = 0; r < m.r; r++) {
        for (bsize c = 0; c < m.c; c++) {
            vec_el(dst, r) += mat_el(m, r, c) * vec_el(v, c);
        }
    }
}