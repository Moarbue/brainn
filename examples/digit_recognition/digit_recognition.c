#include <stdio.h>
#include "../../include/nn.h"

Mat mnist_load(const char *img_file_path, const char *label_file_path, int normalize);
void print_img(Vec img, bsize w, bsize h);

#define IMG_WIDTH  28
#define IMG_HEIGHT 28
#define IMG_RESOLUTION (IMG_WIDTH*IMG_HEIGHT)
#define DIGIT_COUNT 10

#define EPOCHS 100000
#define BATCH_SIZE (28*28)
#define LEARNING_RATE 1e-1

bsize arch[] = {IMG_RESOLUTION, 10, 28, 28, DIGIT_COUNT};
bsize layers = sizeof (arch) / sizeof (arch[0]);

int main(int argc, char **argv)
{
    if (argc != 5) {
        PANIC("Usage: ./digit_recogniton <image training database> <label training database> <image evaluation database> <label evaluation database>");
    }

    Mat data = mnist_load(argv[1], argv[2], 1);

    Mat tinput  = mat_sub_mat(data, 0, 0, data.r, IMG_RESOLUTION);
    Mat toutput = mat_sub_mat(data, 0, IMG_RESOLUTION, data.r, DIGIT_COUNT);

    NN nn = nn_alloc(arch, layers);
    nn_init(nn, -1, 1);
    nn_set_activation_function(&nn, lReLU, dlReLU, Sigmoid, dSigmoid);

    nn_ptrain(&nn, tinput, toutput, BATCH_SIZE, EPOCHS, LEARNING_RATE, 10, 1);

    Mat eval   = mnist_load(argv[3], argv[4], 1);
    Mat input  = mat_sub_mat(data, 0, 0, eval.r, IMG_RESOLUTION);
    Mat output = mat_sub_mat(data, 0, IMG_RESOLUTION, eval.r, DIGIT_COUNT);

    for (bsize r = 0; r < 5; r++) {
        nn_forward(nn, mat_to_row_vec(input, r));
        print_img(mat_to_row_vec(input, r), IMG_WIDTH, IMG_HEIGHT);

        bsize expected;
        for (bsize i = 0; i < DIGIT_COUNT; i++) {
            printf("%zu: %.3f%% ", i, vec_el(nn_output(nn), i) * 100.f);
            if (vec_el(output, i) == 1.f) expected = i;
        }
        printf("\nExpected: %zu\n", expected);
    }

    nn_free(nn);
    mat_free(data);
    mat_free(eval);
}

void print_img(Vec img, bsize w, bsize h)
{
    for (bsize y = 0; y < h; y++) {
        for (bsize x = 0; x < w; x++) {
            bfloat p = vec_el(img, y*w + x);
            if (p == 0) printf("    ");
            else printf("%.3f ", p);
        }
        printf("\n");
    }
}