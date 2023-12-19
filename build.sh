#!/bin/sh

: ${DEBUG:=true}

AR=ar
AR_FLAGS=-rcs
CC=gcc
CFLAGS_DEBUG="-Wall -Wextra -O0 -ggdb3 -g"
CFLAGS_RELEASE="-Wall -Wextra -O3"
CFLAGS=""

LIB_NAME=libbrainn.a
LIB_PATH=lib
BIN_PATH=bin
LIBS="-lm -lpthread -Llib -lbrainn"

if [ "$DEBUG" = true ] ; then   CFLAGS=$CFLAGS_DEBUG
else    CFLAGS=$CFLAGS_RELEASE
fi

set -xe

mkdir -p $LIB_PATH
mkdir -p $BIN_PATH
rm -rf $LIB_PATH/$LIB_NAME

$CC $CFLAGS -c -o $LIB_PATH/mat.o src/mat.c
$CC $CFLAGS -c -o $LIB_PATH/vec.o src/vec.c
$CC $CFLAGS -c -o $LIB_PATH/vec_mat.o src/vec_mat.c
$CC $CFLAGS -c -o $LIB_PATH/nn.o src/nn.c
$CC $CFLAGS -c -o $LIB_PATH/loss.o src/loss.c
$CC $CFLAGS -c -o $LIB_PATH/activation.o src/activation.c
$CC $CFLAGS -c -o $LIB_PATH/nn_parallel.o src/nn_parallel.c
$CC $CFLAGS -c -o $LIB_PATH/nn_io.o src/nn_io.c
$CC $CFLAGS -c -o $LIB_PATH/buffer.o src/buffer.c
$CC $CFLAGS -c -o $LIB_PATH/serialization.o src/serialization.c

$AR -rcs $LIB_PATH/$LIB_NAME $LIB_PATH/mat.o $LIB_PATH/vec.o $LIB_PATH/vec_mat.o $LIB_PATH/nn.o $LIB_PATH/loss.o $LIB_PATH/activation.o $LIB_PATH/nn_parallel.o $LIB_PATH/nn_io.o $LIB_PATH/buffer.o $LIB_PATH/serialization.o

rm -rf $LIB_PATH/mat.o $LIB_PATH/vec.o $LIB_PATH/vec_mat.o $LIB_PATH/nn.o $LIB_PATH/loss.o $LIB_PATH/activation.o $LIB_PATH/nn_parallel.o $LIB_PATH/nn_io.o $LIB_PATH/buffer.o $LIB_PATH/serialization.o

$CC $CFLAGS -o $BIN_PATH/logic_gates examples/logic_gates.c $LIBS
$CC $CFLAGS -o $BIN_PATH/img_compression examples/img_compression/img_compression.c $LIBS
$CC $CFALGS -o $BIN_PATH/digit_recognition examples/digit_recognition/digit_recognition.c examples/digit_recognition/mnist_loader.c $LIBS
$CC $CFLAGS -o $BIN_PATH/io examples/save_load/save_load.c $LIBS