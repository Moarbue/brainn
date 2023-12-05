#include <stdio.h>
#include <stdlib.h>

#include "../../config.h"
#include "../../include/mat.h"

#define IMG_FILE_MAGIC_NUMBER 0x803
#define LBL_FILE_MAGIC_NUMBER 0x801
#define IMG_WIDTH  28
#define IMG_HEIGHT 28
#define IMG_RESOLUTION (IMG_WIDTH*IMG_HEIGHT)
#define DIGIT_COUNT 10

#define bin_to_int(bin) ((int) (bin[0] << 24 | bin[1] << 16 | bin[2] << 8 | bin[3] << 0))

Mat mnist_load(const char *img_file_path, const char *label_file_path, int normalize)
{
    // Open files
    FILE *f_img, *f_lbl;

    f_img = fopen(img_file_path, "rb");
    if (f_img == NULL) PANIC("mnist_load(): Failed to open image file \'%s\'", img_file_path);

    f_lbl = fopen(label_file_path, "rb");
    if (f_lbl == NULL) PANIC("mnist_load(): Failed to open image file \'%s\'", label_file_path);

    // Check magic numbers
    unsigned char bin[4];
    int val;

    fread(bin, sizeof (unsigned char), 4, f_img);
    val = bin_to_int(bin);
    if (val != IMG_FILE_MAGIC_NUMBER) PANIC("mnist_load(): Image file has wrong magic number!");
    printf("INFO: Found image file %s\n", img_file_path);

    fread(bin, sizeof (unsigned char), 4, f_lbl);
    val = bin_to_int(bin);
    if (val != LBL_FILE_MAGIC_NUMBER) PANIC("mnist_load(): Label file has wrong magic number!");
    printf("INFO: Found label file %s\n", label_file_path);

    // Get sample count
    int imgc, lblc;
    
    fread(bin, sizeof (unsigned char), 4, f_img);
    imgc = bin_to_int(bin);

    fread(bin, sizeof (unsigned char), 4, f_lbl);
    lblc = bin_to_int(bin);

    if (imgc != lblc) PANIC("mnist_load(): Image count differs from label count \'%d:%d\'", imgc, lblc);

    printf("INFO: Found %d images inside %s\n", imgc, img_file_path);

    // Get image dimensions
    int w, h;

    fread(bin, sizeof (unsigned char), 4, f_img);
    w = bin_to_int(bin);

    fread(bin, sizeof (unsigned char), 4, f_img);
    h = bin_to_int(bin);

    if (w != IMG_WIDTH || h != IMG_HEIGHT) 
        PANIC("mnist_load(): Images have wrong dimensions \'%dx%d\' instead of \'%dx%d\'", w, h, IMG_WIDTH, IMG_HEIGHT);

    printf("INFO: Images are of size %dx%d\n", w, h);
    
    // Get images
    Mat m = mat_alloc(imgc, w*h + DIGIT_COUNT);   // pixels + label
    unsigned char img[IMG_RESOLUTION];
    unsigned char lbl;

    for (bsize r = 0; r < m.r; r++) {
        fread(img, sizeof (unsigned char), IMG_RESOLUTION, f_img);
        for (bsize c = 0; c < IMG_RESOLUTION; c++) {
            mat_el(m, r, c) = (bfloat) img[c];
            if (normalize) mat_el(m, r, c) /= 255.f;
        }

        fread(&lbl, sizeof (unsigned char), 1, f_lbl);
        for (bsize c = 0; c < DIGIT_COUNT; c++) {
            mat_el(m, r, IMG_RESOLUTION + c) = ((unsigned char)c == lbl);
        }
    }

    printf("INFO: Finished extracting images and labels\n");

    fclose(f_img);
    fclose(f_lbl);

    return m;
}