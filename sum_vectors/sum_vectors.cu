#include <stdio.h>
#include "../common/book.h"
#include <ctime>

#define N 100000000

__global__ void vec_sum(float* a, float* b, float* c) {
	int bid = blockIdx.x;
	if (bid < N) {
		c[bid] = a[bid] + b[bid];
	}
}

int main(void) {

	float *a = (float *)malloc(N * sizeof(float));
	float *b = (float *)malloc(N * sizeof(float));
	float *c = (float *)malloc(N * sizeof(float));

	for (int i = 0; i < N; i++) {
		a[i] = -i * .5;
		b[i] = (i * i) * .25;
	}

	double start_cpu = clock();
	for (int i = 0; i < N; i++) {
		c[i] = a[i] +  b[i];
	}
	double end_cpu = clock();

	printf("cpu time is %f seconds\n", double(end_cpu - start_cpu) / CLOCKS_PER_SEC);

	double start_gpu = clock();
	float *dev_a;
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	float *dev_b;
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
	float *dev_c;
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(float)));

	HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));

	//dim3 grid(N);
	vec_sum<<<N,1>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(float) * N, cudaMemcpyDeviceToHost));

	double end_gpu = clock();
	printf("GPU time: %f seconds\n", (end_gpu - start_gpu) / CLOCKS_PER_SEC);
	printf("a[25] is %f, b[25] is %f, c[25] is %f\n",  a[25], b[25],  c[25]);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	free(a);
	free(b);
	free(c);

	return 0;
}