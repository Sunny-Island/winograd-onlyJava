all:
	gcc -std=c11 -fopenmp -lmkl_rt -liomp5 -O3 driver.c winograd.c -o winograd -mavx512f
	# gcc -std=c11 -D__DEBUG -O0 -g driver.c winograd.c -o winograd
