Implementation of a parallel version of the image normalization algorithm using MPI and OpenMP.

====
Installation
====

		gcc generate_image.c -o generate
		./generate

		mpicc image_normalization.c -fopenmp -o normalization
		mpirun normalization <newMin> <newMax> -np <# of parallel processes>