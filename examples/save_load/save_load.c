#include <stdlib.h>

#include "../../include/nn.h"

bsize arch[] = {2, 2, 1};
bsize layers = sizeof (arch) / sizeof (arch[0]);

int main(void)
{
    NN nn;
    nn = nn_alloc(arch, layers);
    nn_init(nn, -1, 1);

    nn_save("bin/test.nn", nn);
    NN nn2 = nn_load("bin/test.nn");

    nn_print(nn);
    nn_print(nn2);

    nn_free(nn);
    nn_free(nn2);

    return EXIT_SUCCESS;
}