Implementation of a parallel version of the image normalization algorithm using MPI and OpenMP

		gcc generate_image.c -o generate
		./generate

		mpi image_normalization.c -fopenmp -o normalization
		mpirun -np <# of parallel processes> normalization

