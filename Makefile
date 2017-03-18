INC=-I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc
LIB=-I/usr/local/cuda/lib     -L/Developer/GPU\ Computing/C/lib

CFLAGS=-O4
NVCC=--ptxas-options=-v

SRC=src
OUT=.

first: all

all:
	make cpu
	make gpu

cpu: $(SRC)/*.cpp $(SRC)/*.h
	g++ $(CFLAGS) $(LIB) $(INC) $(SRC)/main.cpp -o $(OUT)/cpu

gpu: $(SRC)/*.cpp $(SRC)/*.h
	cp $(SRC)/main.cpp $(SRC)/main.cu
	nvcc $(CFLAGS) $(LIB) $(INC) $(NVCC) $(SRC)/main.cu -o $(OUT)/gpu
	rm $(SRC)/main.cu

clean:
	rm cpu gpu

unsee:
	rm out/*.bmp
