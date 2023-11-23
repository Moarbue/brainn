#include "../include/mat.h"

Mat mat_alloc(bsize r, bsize c)
{
    Mat m;

    m.r = r;
    m.c = c;
    m.e = (bfloat*) BALLOC(r * c * sizeof (bfloat));
    
    if (m.e == NULL) PANIC("Failed to allocate memory for matrix!");

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