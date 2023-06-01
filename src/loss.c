#include "../include/loss.h"

#include <math.h>

float SEL(float t, float x)
{
    return (t - x) * (t - x);
}

float CEL(float t, float x)
{
    x = fmax(x, 1e-7);
    x = fmin(x, 1 - 1e-7);
    return -t * logf(x);
}

