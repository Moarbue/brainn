CC := cc
CFLAGS = -Wall -Wextra
CFLAGS_DEBUG = -O0 -ggdb3 -g
CFLAGS_RELEASE = -O3

AR = ar
AR_FLAGS = -rcs

LIB_PATH = lib
BIN_PATH = bin
SRC_PATH = src
LIB_NAME = libbrainn.a
LIB_OBJ  = $(LIB_PATH)/activation.o $(LIB_PATH)/loss.o $(LIB_PATH)/mat.o $(LIB_PATH)/nn_io.o $(LIB_PATH)/nn_parallel.o $(LIB_PATH)/nn.o $(LIB_PATH)/vec_mat.o $(LIB_PATH)/vec.o 
LIBS = -lm -lpthread -L$(LIB_PATH) -lbrainn

EXAMPLE_DIR = examples
LOGIC_GATES_SRC = $(EXAMPLE_DIR)/logic_gates.c
IMAGE_COMPRESSION_SRC = $(EXAMPLE_DIR)/img_compression/img_compression.c
DIGIT_RECOGNITION_SRC = $(EXAMPLE_DIR)/digit_recognition/digit_recognition.c $(EXAMPLE_DIR)/digit_recognition/mnist_loader.c
SAVE_LOAD_SRC = $(EXAMPLE_DIR)/save_load/save_load.c

.PHONY=all release debug mkdirs build_lib build_examples logic_gates image_compression digit_recognition save_load clean

all: release

release: CFLAGS += $(CFLAGS_RELEASE)
release: build_lib build_examples

debug: CFLAGS += $(CFLAGS_DEBUG)
debug: build_lib build_examples

mkdirs:
	mkdir -p $(LIB_PATH)
	mkdir -p $(BIN_PATH)

$(LIB_PATH)/%.o: $(SRC_PATH)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

build_lib: mkdirs $(LIB_OBJ)
	$(AR) $(AR_FLAGS) $(LIB_PATH)/$(LIB_NAME) $(LIB_OBJ)
	rm $(LIB_OBJ)


build_examples:	logic_gates image_compression digit_recognition save_load

logic_gates:
	$(CC) $(CFLAGS) -o $(BIN_PATH)/logic_gates $(LOGIC_GATES_SRC) $(LIBS)

image_compression:
	$(CC) $(CFLAGS) -o $(BIN_PATH)/image_compression $(IMAGE_COMPRESSION_SRC) $(LIBS)

digit_recognition:
	$(CC) $(CFLAGS) -o $(BIN_PATH)/digit_recognition $(DIGIT_RECOGNITION_SRC) $(LIBS)

save_load:
	$(CC) $(CFLAGS) -o $(BIN_PATH)/save_load $(SAVE_LOAD_SRC) $(LIBS)

clean:
	rm -rf $(LIB_PATH) $(BIN_PATH)
	