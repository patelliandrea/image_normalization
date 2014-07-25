#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <time.h>
#include <omp.h>
static int x;
static int y;

#define BENCH_GETTIME(x) do {		\
	gettimeofday((x), NULL);	\
      } while(0)

#include <sys/time.h>
void print_duration(struct timeval start, struct timeval end) {
    double duration = ((end.tv_sec-start.tv_sec)*1000000
            + end.tv_usec - start.tv_usec)/1000.0;
    fprintf(stderr, "duration = %lf\n", duration);
}


unsigned char *readPPM(const char *filename) {
	char buff[16];
	unsigned char *result;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	fgets(buff, sizeof(buff), fp);
	c = getc(fp);
	while(c == '#') {
		while(getc(fp) != '\n') ;
		c = getc(fp);
	}

	ungetc(c, fp);

	fscanf(fp, "%d %d", &x, &y);

	fscanf(fp, "%d", &rgb_comp_color);

	while(fgetc(fp) != '\n') ;
	result = (unsigned char *)malloc(3 * x * y * sizeof(unsigned char));
	fread(result, 3 * x, y, fp);
	fclose(fp);	
	return result;
}

void writePPM(const char *filename, unsigned char *image) {
	FILE *fp;
	fp = fopen(filename, "wb");
	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", x, y);
	fprintf(fp, "%d\n", 255);
	fwrite(image, 3 * x, y, fp);
	fclose(fp);
}

void findMinMax(unsigned char *img, unsigned char *min, unsigned char *max, int size) {
	long i;
	unsigned char value;

	unsigned char *localMin = malloc(3 * sizeof(unsigned char));
	unsigned char *localMax = malloc(3 * sizeof(unsigned char));

	for(i = 0; i < 3; i++) {
		localMin[i] = 255;
		localMax[i] = 0;
	}

	// the computation is divided among threads, each calculates localMin and localMax
	#pragma omp parallel firstprivate(localMin, localMax) private(value) shared(img, size, min, max)
	{
		#pragma omp for
		for(i = 0; i < size; i++) {
			value = img[i];
			if(value < localMin[i % 3]) {
				localMin[i % 3] = value;
			}
			if(value > localMax[i % 3]) {
				localMax[i % 3] = value;
			}
		}

		// localMin and localMax are reduced into global min and max
		for(i = 0; i < 3; i++) {
			#pragma omp critical
			{
				min[i] = min[i] < localMin[i] ? min[i] : localMin[i];
				max[i] = max[i] > localMax[i] ? max[i] : localMax[i];
			}
		}
	}

	free(localMin);
	free(localMax);
}

void normalize(unsigned char *img, int newMin, int newMax, unsigned char *min, unsigned char *max, int size) {
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	long i;

	for(i = 0; i < 3; i++) {
		if(max[i] == min[i]) {
			return;
		}
	}

	if(img) {
		// every thread normalizes its part of the image
		#pragma omp parallel for shared(img) firstprivate(min, max, newMin, newMax)
		for(i = 0; i < size; i++) {
			int j = i % 3;
			img[i] = (img[i] - min[j]) * (newMax - newMin) / (max[j] - min[j]) + newMin;
		}
	}
}

int main(int argc, char *argv[]) {
	// the image to normalize
	unsigned char *image = readPPM("test.ppm");
	int numtasks, rank;
	int *displs;
	int *recvcounts;
	int *sendcounts;

	// MPI initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	// min size of the arrays and extra pixel to allocate
	int nmin = x * y * 3 / numtasks;
	int nextra = x * y * 3 % numtasks;

	// integer array (of length numtasks)
	// Entry i specifies the displacement (relative to sendbuf from which to take the outgoing data to process i
	displs = (int *)malloc(numtasks * sizeof(int));
	// integer array (of lenght numtasks)
	// Entry i specifies the number of elements in receive buffer of the process i
	recvcounts = (int *)malloc(numtasks * sizeof(int));
	// integer array (of lenght numtasks)
	// Entry i specifies the number of elements in send buffer of the process i
	sendcounts = (int *)malloc(numtasks * sizeof(int));

	long i, k;
	// initialization of the three previous arrays
	for(i = 0; i < numtasks; i++) {
		if(i < nextra) {
			sendcounts[i] = recvcounts[i] = nmin + 1;
		} else {
			sendcounts[i] = recvcounts[i] = nmin;
		}
		displs[i] = k;
		k += sendcounts[i];
	}

	// The main process is process 0
	int source = 0;

	// every process allocates its receive buffer
	unsigned char *recvBuffer = (unsigned char *)malloc(recvcounts[rank] * sizeof(unsigned char));
	// the image is scattered in parts to all processes in the communicator MPI_COMM_WORLD
	MPI_Scatterv(image, sendcounts, displs, MPI_UNSIGNED_CHAR, recvBuffer, recvcounts[rank], MPI_UNSIGNED_CHAR, source, MPI_COMM_WORLD);


	// global min and max (for each RGB component of the image)
	unsigned char *min = (unsigned char *)malloc(3 * sizeof(unsigned char));
	unsigned char *max = (unsigned char *)malloc(3 * sizeof(unsigned char));

	// local min and max (for each RGB component of the image) calculated by a single process
	unsigned char *localMin = (unsigned char *)malloc(3 * sizeof(unsigned char));
	unsigned char *localMax = (unsigned char *)malloc(3 * sizeof(unsigned char));

	// initialization of localMin and localMax
	for(i = 0; i < 3; i++) {
		localMin[i] = 255;
		localMax[i] = 0;
	}
	struct timeval tStart, tEnd;

	// every process finds the min and max of its part of the image
	BENCH_GETTIME(&tStart);
	findMinMax(recvBuffer, localMin, localMax, recvcounts[rank]);
	BENCH_GETTIME(&tEnd);
	print_duration(tStart, tEnd);

	// localMin and localMax are reduced into min and max
	MPI_Allreduce(localMin, min, 3, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(localMax, max, 3, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);

	// every process normalizes its parts of the image
	BENCH_GETTIME(&tStart);
	normalize(recvBuffer, atoi(argv[1]), atoi(argv[2]), min, max, recvcounts[rank]);
	BENCH_GETTIME(&tEnd);
	print_duration(tStart, tEnd);

	// every normalized part is gathered into image
	MPI_Gatherv(recvBuffer, sendcounts[rank], MPI_UNSIGNED_CHAR, image, recvcounts, displs, MPI_UNSIGNED_CHAR, source, MPI_COMM_WORLD);

	// the main process saves the image
	if(rank == 0) {
		writePPM("result.ppm", image);
	}

	// free allocated memory
	free(min);
	free(max);
	free(localMin);
	free(localMax);
	free(recvBuffer);
	free(displs);
	free(recvcounts);
	free(sendcounts);
	free(image);
	MPI_Finalize();
}