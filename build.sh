#!/bin/sh

CFLAGS="-O3 -Wall -Wextra"

set -xe

mkdir -p lib
gcc $CFLAGS -c src/vec.c         -o lib/vec.o
gcc $CFLAGS -c src/mat.c         -o lib/mat.o
gcc $CFLAGS -c src/vec_mat.c     -o lib/vec_mat.o
gcc $CFLAGS -c src/activation.c  -o lib/activation.o
gcc $CFLAGS -c src/loss.c        -o lib/loss.o
gcc $CFLAGS -c src/nn.c          -o lib/nn.o
gcc $CFLAGS -c src/optimizer.c   -o lib/optimizer.o

ar -rcs lib/libbrainn.a lib/vec.o lib/mat.o lib/vec_mat.o lib/activation.o lib/loss.o lib/nn.o lib/optimizer.o

rm -rf lib/vec.o lib/mat.o lib/vec_mat.o lib/activation.o lib/loss.o lib/nn.o