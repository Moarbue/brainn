#include <stdio.h>
#include <time.h>

#include "../include/nn.h"

#define EPOCHS 20000
#define BATCH_SIZE 200
#define LEARNING_RATE 1

Mat or_gate = {
    .r = 4,
    .c = 3,
    .s = 0,
    .e = (bfloat[]) {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 1,
    },
};

Mat and_gate = {
    .r = 4,
    .c = 3,
    .s = 0,
    .e = (bfloat[]) {
        0, 0, 0,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    },
};

Mat nor_gate = {
    .r = 4,
    .c = 3,
    .s = 0,
    .e = (bfloat[]) {
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        1, 1, 0,
    },
};

Mat nand_gate = {
    .r = 4,
    .c = 3,
    .s = 0,
    .e = (bfloat[]) {
        0, 0, 1,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    },
};

Mat xor_gate = {
    .r = 4,
    .c = 3,
    .s = 0,
    .e = (bfloat[]) {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    },
};

Mat xnor_gate = {
    .r = 4,
    .c = 3,
    .s = 0,
    .e = (bfloat[]) {
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    },
};

int main(void)
{
    srand(time(0));

    bsize arch[] = {2, 2, 1};
    bsize layers = sizeof (arch) / sizeof (arch[0]);

    NN nn = nn_alloc(arch, layers);
    nn_init(nn, 0, 0.01);

    Mat data   = xnor_gate;
    bsize s    = data.r;
    Mat input  = mat_sub_mat(data, 0, 0, s, 2);
    Mat output = mat_sub_mat(data, 0, 2, s, 1);

    for (bsize e = 0; e < EPOCHS; e++) {
        for (bsize b = 0; b < BATCH_SIZE; b++) {
            nn_forward(nn, mat_to_row_vec(input, b % s));
            nn_backpropagate(nn, mat_to_row_vec(output, b % s));
        }
        nn_evolve(nn, LEARNING_RATE);
        printf("E: %zu  L: %.5f\r", e, nn_loss(nn, input, output));
    }
    printf("\n");

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            nn_forward(nn, mat_to_row_vec(input, 2*i + j));
            printf("%zu | %zu = %.3f\n", i, j, vec_el(nn_output(nn), 0));
        }
    }

    nn_free(nn);
    return 0;
}