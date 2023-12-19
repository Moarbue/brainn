#include "../include/buffer.h"
#include "../config.h"

#include <errno.h>
#include <string.h>

Buffer *buffer_alloc(void)
{
    Buffer *b;
    b = (Buffer *) BALLOC(sizeof (Buffer));
    if (b == NULL) PANIC("buffer_alloc(): Failed to allocate memory!");

    b->d = (uint8_t *) BALLOC(INITIAL_BUFFER_SIZE);
    b->c = INITIAL_BUFFER_SIZE;
    b->s = 0;
    b->o = 0;

    return b;
}

void buffer_add(Buffer *b, uint8_t *d, size_t s)
{
    while (b->s + s > b->c) {
        b->c *= 2;
        b->d = (uint8_t *) BREALLOC(b->d, b->c);
    }

    memcpy(b->d + b->s, d, s);
    b->s += s;
}

uint8_t *buffer_advance(Buffer *b, size_t c)
{
    uint8_t *res = b->d + b->o;
    b->o += c;

    return res;
}

Buffer *buffer_from_file(FILE *f)
{
    uint8_t *d;
    size_t s;

    fseek(f, 0, SEEK_END);
    s = ftell(f);
    fseek(f, 0, SEEK_SET);

    d = (uint8_t *) BALLOC(s * sizeof (uint8_t));
    if (d == NULL) PANIC("buffer_read(): Failed to allocate memory!");

    if (fread(d, sizeof (uint8_t), s, f) != s) 
        PANIC("buffer_from_file(): Failed to read from file: %s", strerror(errno));
    
    Buffer *b;
    b = buffer_alloc();
    buffer_add(b, d, s);

    free(d);
    return b;
}

void buffer_write(Buffer *b, FILE *f)
{
    if (fwrite(b->d, 1, b->s, f) != b->s)
        PANIC("buffer_write(): Failed to write buffer to file: %s", strerror(errno));
}

void buffer_free(Buffer *b)
{
    free(b->d);
    free(b);
}
