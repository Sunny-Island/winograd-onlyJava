all:
	gcc -std=c11 -fopenmp -O3 driver.c winograd.c -o winograd -mfma
	# gcc -std=c11 -D__DEBUG -O0 -g driver.c winograd.c -o winograd
