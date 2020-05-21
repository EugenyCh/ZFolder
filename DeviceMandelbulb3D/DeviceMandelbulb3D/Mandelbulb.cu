#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Mandelbulb.cuh"
#include "Hypercomplex.cuh"
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (b) : (a))

__global__ void kernel(
	byte* buffer,
	const int side,
	const int n,
	const int maxIter,
	const float bailout,
	const float sqrBailout)
{
	int offset = threadIdx.x + blockDim.x * blockIdx.x;
	int z = offset / (side * side);
	int y = (offset - z) / side;
	int x = offset % side;
	if (z >= side)
		return;

	// Compute point at this position
	int halfSide = side >> 1;
	float fx = bailout * (float)(x - halfSide) / halfSide;
	float fy = bailout * (float)(y - halfSide) / halfSide;
	float fz = bailout * (float)(z - halfSide) / halfSide;
	Hypercomplex hc(fx, fy, fz);
	Hypercomplex hz(fx, fy, fz);

	// Iterating
	bool belongs;
	if (hc.sqrRadius() > sqrBailout)
		belongs = false;
	else
	{
		for (int i = 0; i < maxIter; ++i)
			hz = (hz ^ n) + hc;
		belongs = hz.sqrRadius() <= sqrBailout;
	}

	if (belongs)
		buffer[offset] = (byte)((hz.sqrRadius() / sqrBailout) * 255);
	else
		buffer[offset] = 0;
}

Mandelbulb::Mandelbulb(int power, int maxIter)
{
	this->n = power;
	this->maxIter = maxIter;
	this->bailout = powf(2.0f, 1.0f / (power - 1));
	this->sqrBailout = powf(4.0f, 1.0f / (power - 1));
}

bool Mandelbulb::compute(size_t width, size_t height)
{
	if (points)
		delete[] points;
	side = MIN(width, height);

	const size_t sz = side * side * side;
	points = new byte[sz];

	printf("Computing %d^3 points\n", side);
	// Processing
	byte* dev_buffer;

	if (cudaMalloc((void**)&dev_buffer, sz) != cudaSuccess)
	{
		printf("Error on creating buffer of pixels in GPU\n");
		return false;
	}

	clock_t tStart, tFinish;
	double tDelta;
	printf("Rendering %dx%d\n", width, height);
	int threads = 1024;
	int blocks = (sz + threads - 1) / threads;
	tStart = clock();
	kernel<<<blocks, threads>>>(dev_buffer, side, n, maxIter, bailout, sqrBailout);
	cudaThreadSynchronize();
	tFinish = clock();
	tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("It tooks %.3f seconds\n", tDelta);

	printf("Moving\n");
	if (cudaMemcpy((void*)points, dev_buffer, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error on getting buffer of pixels from GPU\n");
		return false;
	}
	cudaFree(dev_buffer);

	printf("\nCeaning of points\n");
	tStart = clock();
	int* pointsToCleaning = new int[sz];
	int cleaned = 0;
	int index = 0;
	for (int z = 1; z < side - 1; ++z)
	{
		for (int y = 1; y < side - 1; ++y)
		{
			for (int x = 1; x < side - 1; ++x)
			{
				int offset000 = (z - 1) * side * side + (y - 1) * side + (x - 1);
				int offset001 = (z - 1) * side * side + (y - 1) * side + (x + 1);
				int offset010 = (z - 1) * side * side + (y + 1) * side + (x - 1);
				int offset011 = (z - 1) * side * side + (y + 1) * side + (x + 1);
				int offset100 = (z + 1) * side * side + (y - 1) * side + (x - 1);
				int offset101 = (z + 1) * side * side + (y - 1) * side + (x + 1);
				int offset110 = (z + 1) * side * side + (y + 1) * side + (x - 1);
				int offset111 = (z + 1) * side * side + (y + 1) * side + (x + 1);
				bool h000 = points[offset000] > 0;
				bool h001 = points[offset001] > 0;
				bool h010 = points[offset010] > 0;
				bool h011 = points[offset011] > 0;
				bool h100 = points[offset100] > 0;
				bool h101 = points[offset101] > 0;
				bool h110 = points[offset110] > 0;
				bool h111 = points[offset111] > 0;
				if (h000 && h001 && h010 && h011 && h100 && h101 && h110 && h111)
				{
					int offset = z * side * side + y * side + x;
					pointsToCleaning[index++] = offset;
					++cleaned;
				}
			}
		}
	}
	for (int i = 0; i < index; ++i)
		points[pointsToCleaning[i]] = 0;
	// End
	tFinish = clock();
	tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("\nIt tooks %.3f seconds\n", tDelta);
	return true;
}

void Mandelbulb::draw(size_t width, size_t height)
{
	if (points)
	{
		glBegin(GL_POINTS);
		int shiftX = (width - side) / 2 - width / 2;
		int shiftY = (height - side) / 2 - height / 2;
		int shiftZ = MAX(shiftX, shiftY);
		for (int z = 0; z < side; ++z)
		{
			for (int y = 0; y < side; ++y)
			{
				for (int x = 0; x < side; ++x)
				{
					int i = z * side * side + y * side + x;
					if (points[i] > 0)
					{
						float k = points[i] / 255.0f;
						byte kBlue = (byte)(k * 255);
						byte kGreen = (byte)((k / 2) * 255);
						byte kRed = (byte)((k / 3) * 255);
						glColor3ub(
							kRed,
							kGreen,
							kBlue
						);
						glVertex3f(
							shiftX + x,
							shiftY + y,
							shiftZ + z
						);
					}
				}
			}
		}
		glEnd();
	}
}
