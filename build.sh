#!/bin/sh

DEBUG=true

AR=ar
AR_FLAGS=-rcs
CC=gcc
CFLAGS_DEBUG="-Wall -Wextra -O0 -ggdb -g"
CFLAGS_RELEASE="-Wall -Wextra -O3"
CFLAGS=""

LIB_NAME=libbrainn.a
LIB_PATH=lib
BIN_PATH=bin
LIBS="-lm -Llib -lbrainn"

if [ "$DEBUG" = true ] ; then   CFLAGS=$CFLAGS_DEBUG
else    CFLAGS=$CFLAGS_RELEASE
fi

set -xe

mkdir -p $LIB_PATH
mkdir -p $BIN_PATH

$CC $CFLAGS -c -o $LIB_PATH/mat.o src/mat.c
$CC $CFLAGS -c -o $LIB_PATH/vec.o src/vec.c
$CC $CFLAGS -c -o $LIB_PATH/vec_mat.o src/vec_mat.c

$AR -rcs $LIB_PATH/$LIB_NAME $LIB_PATH/mat.o $LIB_PATH/vec.o $LIB_PATH/vec_mat.o

rm -rf $LIB_PATH/mat.o $LIB_PATH/vec.o $LIB_PATH/vec_mat.o

$CC $CFLAGS -o $BIN_PATH/logic_gates examples/logic_gates.c $LIBS