all: main.o
main.o: main.cu matrixTranspose.cu matrixMul.cu
	nvcc main.cu  matrixTranspose.cu matrixMul.cu -o main.o -lcublas
clean:
	rm *.o
