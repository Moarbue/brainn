#include "../include/mat.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.r = rows;
    m.c = cols;
    m.s = 0;
    m.e = (float *) malloc(sizeof (*m.e) * rows * cols);

    assert(m.e != NULL && "Failed to allocate memory for matrix!");

    return m;
}

void mat_fill(Mat m, float val)
{
    assert(m.e != NULL && "No memory for elements allocated!");

    for (size_t r = 0; r < m.r; r++) {
        for (size_t c = 0; c < m.c; c++) {
            mat_el(m, r, c) = val;
        }
    }
}

void mat_rand(Mat m, float min, float max)
{
    assert(m.e != NULL && "No memory for elements allocated!");

    for (size_t r = 0; r < m.r; r++) {
        for (size_t c = 0; c < m.c; c++) {
            mat_el(m, r, c) = ((float) rand() / (float) RAND_MAX) * (max - min) + min;
        }
    }
}

Mat mat_sub_mat(Mat m, size_t rows, size_t cols, size_t row_offset, size_t col_offset)
{
    assert(m.r >= rows && "Sub-Matrix rows are out of bounds");
    assert(m.c >= cols && "Sub-Matrix columns are out of bounds");
    assert(m.r > row_offset && "row-offset is out of bounds");
    assert(m.c > col_offset && "column-offset is out of bounds");

    Mat dst;
    dst.r = rows;
    dst.c = cols;
    dst.s = m.c - cols + m.s;
    dst.e = &mat_el(m, row_offset, col_offset);

    return dst;
}

void mat_free(Mat m)
{
    free(m.e);
}


void mat_print_intern(Mat m, const char *name, size_t pad)
{
    assert(m.e != NULL && "No memory for elements allocated!");

    printf("%*s%s = [\n", (int) pad, "", name);
    for (size_t r = 0; r < m.r; r++) {
        printf("%*s    ", (int) pad, "");
        for (size_t c = 0; c < m.c; c++) {
            printf("%.3f ", mat_el(m, r, c));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) pad, "");
}
