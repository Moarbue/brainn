#include "../include/vec_mat.h"

#include <assert.h>

void vec_mat_mul(Vec dst, Vec v, Mat m)
{
    assert(dst.e != NULL && "No memory for elements of dst allocated!");
    assert(v.e   != NULL && "No memory for elements of v allocated!");
    assert(m.e   != NULL && "No memory for elements of m allocated!");

    assert(v.c   == m.r);
    assert(dst.c == m.c);

    for (size_t r = 0; r < dst.c; r++) {
        for (size_t c = 0; c < v.c; c++) {
            vec_el(dst, r) += vec_el(v, c) * mat_el(m, c, r);
        }
    }
}

Vec mat_to_row_vec(Mat m, size_t row)
{
    assert(row < m.r && "Row index is outside matrix bounds");

    Vec dst;
    dst.c = m.c;
    dst.s = 0;
    dst.e = &mat_el(m, row, 0);

    return dst;
}

Vec mat_to_col_vec(Mat m, size_t col)
{
    assert(col < m.c && "Column index is outside matrix bounds");

    Vec dst;
    dst.c = m.r;
    dst.s = m.c;
    dst.e = &mat_el(m, 0, col);

    return dst;
}
