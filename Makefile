all:
	gcc -std=c11 -fopenmp -O2 driver.c winograd.c -o winograd
	# gcc -std=c11 -D__DEBUG -O0 -g driver.c winograd.c -o winograd
