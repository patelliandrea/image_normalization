#include<stdio.h>
#include<stdlib.h>

int main() {
	unsigned char *image = (unsigned char *)malloc(3 * 10000 * 10000 * sizeof(unsigned char));

	unsigned long i;
	for(i = 0; i < 3 * 10000 * 10000; i++) {
		image[i] = rand() % 256;
	}
	FILE *fp;
	fp = fopen("test.ppm", "wb");
	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n", 10000, 30000);
	fprintf(fp, "%d\n", 255);
	fwrite(image, 3 * 10000, 10000, fp);

	for(i = 0; i < 3 * 10000 * 10000; i++) {
		image[i] = rand() % 256;
	}
	fwrite(image, 3 * 10000, 10000, fp);

	for(i = 0; i < 3 * 10000 * 10000; i++) {
		image[i] = rand() % 256;
	}
	fwrite(image, 3 * 10000, 10000, fp);
	
	fclose(fp);
}