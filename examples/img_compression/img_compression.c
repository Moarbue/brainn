#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <time.h>

#include "../../include/nn.h"

#define IMG_PATH "examples/img_compression/8.png"
#define IMG_CHANNELS 1
#define IMG_RESOLUTION (28*28)
#define OUT_PATH "bin/out.png"
#define OUT_WIDTH  420
#define OUT_HEIGHT 420
#define OUT_RESOLUTION (OUT_WIDTH*OUT_HEIGHT)

#define EPOCHS 100000
#define BATCH_SIZE (28)
#define LEARNING_RATE 1e-1

bsize arch[] = {2, 28, 28, 9, 1};
bsize layers = sizeof (arch) / sizeof (arch[0]);

Vec img_to_vec(const char *file_name, bsize *width, bsize *height);
void vec_to_img(Vec img, bsize w, bsize h, const char *file_name);
void print_img(Vec img, bsize w, bsize h);

int main(void)
{
    srand(time(0));

    bsize w, h;
    Vec img = img_to_vec(IMG_PATH, &w, &h);
    print_img(img, w, h);

    Mat data = mat_alloc(w*h, 3);
    for (bsize r = 0; r < data.r; r++) {
        mat_el(data, r, 0) = (bfloat)(r / w) / (bfloat)(h-1);
        mat_el(data, r, 1) = (bfloat)(r % h) / (bfloat)(w-1);
        mat_el(data, r, 2) = vec_el(img, r);
    }

    Mat tinput  = mat_sub_mat(data, 0, 0, data.r, 2);
    Mat toutput = mat_sub_mat(data, 0, 2, data.r, 1);

    vec_free(img);

    NN nn = nn_alloc(arch, layers);
    nn_init(nn, -1, 1);
    nn_set_activation_function(&nn, lReLU, dlReLU, Sigmoid, dSigmoid);
    nn_set_optimizer(&nn, optimizer_SGD(LEARNING_RATE));

    nn_train(nn, tinput, toutput, BATCH_SIZE, EPOCHS, 1);

    img = vec_alloc(OUT_RESOLUTION);
    for (bsize i = 0; i < img.c; i++) {
        bfloat y = ((bfloat) i / (bfloat) OUT_WIDTH) * (bfloat)h / (bfloat)OUT_HEIGHT / (bfloat)(h-1);
        bfloat x = (i % OUT_WIDTH) * (bfloat)w / (bfloat)OUT_WIDTH / (bfloat)(w-1);

        Vec input = {.c = 2, .s = 1, .e = (bfloat[]) {y, x}};
        vec_el(img, i) = vec_el(nn_forward(nn, input), 0);
    }

    vec_to_img(img, OUT_WIDTH, OUT_HEIGHT, OUT_PATH);

    mat_free(data);
    nn_free(nn);
    vec_free(img);
}

Vec img_to_vec(const char *file_name, bsize *width, bsize *height)
{
    stbi_uc *img_raw;
    int w, h, ch;
    img_raw = stbi_load(file_name, &w, &h, &ch, 0);

    if (img_raw == NULL)         PANIC("Failed to load image");
    if (ch != IMG_CHANNELS)      PANIC("Only 8-Bit grayscaled images are supported");
    if (w * h != IMG_RESOLUTION) PANIC("Only 28x28 images are supported");

    Vec img = vec_alloc(w*h);
    for (bsize i = 0; i < img.c; i++) vec_el(img, i) = img_raw[i] / 255.f;

    *width  = w;
    *height = h;    

    stbi_image_free(img_raw);

    return img;
}

void vec_to_img(Vec img, bsize w, bsize h, const char *file_name)
{
    if (w*h != img.c) PANIC("Image dimensions are different form vector length");

    stbi_uc *img_raw = (stbi_uc *) malloc(w * h);

    for (bsize i = 0; i < img.c; i++) {
        img_raw[i] = (stbi_uc) (vec_el(img, i) * 255);
    }

    stbi_write_png(file_name, w, h, 1, img_raw, 0);

    free(img_raw);
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