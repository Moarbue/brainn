#include "../include/mat.h"

Mat mat_alloc(bsize r, bsize c)
{
    Mat m;

    m.r = r;
    m.c = c;
    m.s = 0;
    m.e = (bfloat*) BALLOC(r * c * sizeof (bfloat));
    
    if (m.e == NULL) PANIC("mat_alloc(): Failed to allocate memory!");

    return m;
}

void mat_fill(Mat m, bfloat value)
{
    for (bsize r = 0; r < m.r; r++) {
        for (bsize c = 0; c < m.c; c++) {
            mat_el(m, r, c) = value;
        }
    }
}

void mat_rand(Mat m, bfloat min, bfloat max)
{
    for (bsize r = 0; r < m.r; r++) {
        for (bsize c = 0; c < m.c; c++) {
            mat_el(m, r, c) = brand(min, max);
        }
    }
}

Mat mat_sub_mat(Mat src, bsize r_start, bsize c_start, bsize r, bsize c)
{
    if (r_start >= src.r)        PANIC("mat_sub_mat(): row index is out of bounds");
    if (c_start >= src.c)        PANIC("mat_sub_mat(): col index is out of bounds");
    if (r > src.r - r_start) PANIC("mat_sub_mat(): row size is greater than src size");
    if (c > src.c - c_start) PANIC("mat_sub_mat(): col size is greater than src size");

    Mat dst;

    dst.r = r;
    dst.c = c;
    dst.s = src.c - c;
    dst.e = &mat_el(src, r_start, c_start);

    return dst;
}

void mat_free(Mat m)
{
    BFREE(m.e);
}


void mat_print_intern(Mat m, const char *name, bsize pad)
{
    printf("%*s%s = [\n", (int) pad, "", name);
    for (bsize r = 0; r < m.r; r++) {
        printf("%*s    ", (int) pad, "");
        for (bsize c = 0; c < m.c; c++) {
            printf("%.3f ", mat_el(m, r, c));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) pad, "");
}