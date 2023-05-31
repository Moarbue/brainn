#include "../include/mat.h"

#include <assert.h>
#include <stdlib.h>

Mat  mat_alloc(size_t rows, size_t cols)
{
    Mat m;

    m.r = rows;
    m.c = cols;
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

void mat_free(Mat m)
{
    free(m.e);
}
