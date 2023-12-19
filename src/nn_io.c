#include "../include/nn.h"
#include "../include/serialization.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

char MAGIC_NUMBER[] = "BRAINN";
bsize MAGIC_LENGTH = sizeof (MAGIC_NUMBER) - 1;

FILE *open_file(const char *filename, const char *mode)
{
    FILE *f = fopen(filename, mode);
    if (f == NULL)
        PANIC("open_file(): Failed to open file \'%s\': %s", filename, strerror(errno));

    return f;
}

void nn_save(const char *filename, NN nn)
{
    FILE *fout;
    fout = open_file(filename, "wb");

    Buffer *b;
    b = buffer_alloc();

    serialize_string(b, MAGIC_NUMBER, MAGIC_LENGTH);
    serialize_nn(b, nn);

    buffer_write(b, fout);
    buffer_free(b);

    fclose(fout);
}

NN nn_load(const char *filename)
{
    FILE *fin;
    fin = open_file(filename, "rb");

    Buffer *b;
    b = buffer_from_file(fin);

    if (strncmp(deserialize_string(b, MAGIC_LENGTH), MAGIC_NUMBER, MAGIC_LENGTH) != 0)
        PANIC("nn_load(): \'%s\' is not a valid neural network file!", filename);

    NN nn;
    nn = deserialize_nn(b);

    buffer_free(b);
    fclose(fin);

    return nn;
}