compile:
	nvcc -o heat_cuda heat.cu
	
compile_prod:
	nvcc -U CHECK_CUDA_ERRORS heat.cu -o heat_cuda

sample: compile
	./heat_cuda 2000 10000 sample.bmp

test:
	python test.py

all: clear compile_prod
	python run.py -r ./results.csv

clear:
	rm -fr ./results/* results.csv
