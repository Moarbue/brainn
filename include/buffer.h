#include <inttypes.h>
#include <stdio.h>

#ifndef _BRAINN_BUFFER_H_
#define _BRAINN_BUFFER_H_

#define INITIAL_BUFFER_SIZE 32

typedef struct {
    uint8_t *d;
    size_t o;
    size_t s;
    size_t c;
} Buffer;

Buffer *buffer_alloc(void);
void buffer_add(Buffer *b, uint8_t *d, size_t s);
uint8_t *buffer_advance(Buffer *b, size_t c);
Buffer *buffer_from_file(FILE *f);
void buffer_write(Buffer *b, FILE *f);
void buffer_free(Buffer *b);

#endif // _BRAINN_BUFFER_H_