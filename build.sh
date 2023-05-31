#!/bin/sh

CFLAGS="-O3 -Wall -Wextra"

set -xe

mkdir -p lib
gcc $CFLAGS -c src/vec.c         -o lib/vec.o
gcc $CFLAGS -c src/mat.c         -o lib/mat.o
gcc $CFLAGS -c src/activation.c  -o lib/activation.o
gcc $CFLAGS -c src/nn.c          -o lib/nn.o

ar -rcs lib/libbrainn.a lib/vec.o lib/mat.o lib/activation.o lib/nn.o

rm -rf lib/vec.o lib/mat.o lib/nn.o