#include "../include/loss.h"
#include <math.h>

bfloat SEL(bfloat t, bfloat y)
{
    return (t - y) * (t - y);
}

bfloat CEL(bfloat t, bfloat y)
{
    y = fmax(y, 1e-7);
    y = fmin(y, 1.0 - 1e-7);
    return -t * logf(y);
}


bfloat dSEL(bfloat t, bfloat y)
{
    return 2.0 * (y - t); 
}

bfloat dCEL(bfloat t, bfloat y)
{
    y = fmax(y, 1e-7);
    y = fmin(y, 1.0 - 1e-7);
    return -t / y;
}