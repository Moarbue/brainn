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

void mat_shuffle_rows(Mat m)
{
    for (bsize r = 0; r < m.r-1; r++) 
    {
        size_t j = brand(r+1, m.r - r); // works only because decimals are ignored
        for (bsize c = 0; c < m.c; c++) {
            bfloat t = mat_el(m, j, c);
            mat_el(m, j, c) = mat_el(m, r, c);
            mat_el(m, r, c) = t;
        }
    }
}

Mat mat_sub_mat(Mat src, bsize r_start, bsize c_start, bsize r, bsize c)
{
    if (r_start >= src.r)        PANIC("mat_sub_mat(): row index is out of bounds [%zu] > %zu", r_start, src.r);
    if (c_start >= src.c)        PANIC("mat_sub_mat(): col index is out of bounds [%zu] > %zu", c_start, src.c);
    if (r > src.r - r_start) PANIC("mat_sub_mat(): row size is greater than src size %zu > %zu", r, src.r - r_start);
    if (c > src.c - c_start) PANIC("mat_sub_mat(): col size is greater than src size %zu > %zu", c, src.c - c_start);

    Mat dst;

    dst.r = r;
    dst.c = c;
    dst.s = src.c - c + src.s;
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