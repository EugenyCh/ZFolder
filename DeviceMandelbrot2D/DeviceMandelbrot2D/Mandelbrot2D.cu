#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Mandelbrot2D.cuh"
#include "cuComplex.cuh"
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <stdio.h>
#include <time.h>

__global__ void kernel(byte* buffer, const int side, const float sqrBailout, const float p, const int iters)
{
	int offset = threadIdx.x + blockDim.x * blockIdx.x;
	if (offset >= side * side)
		return;
	int x = offset % side;
	int y = offset / side;

	// Compute point at this position
	int halfSide = side >> 1;
	float jx = 2.0f * (float)(x - halfSide) / halfSide;
	float jy = 2.0f * (float)(y - halfSide) / halfSide;
	//jx -= 0.5f;
	cuComplex c(jx, jy);
	cuComplex z(jx, jy);

	// Iterating
	int i;
	for (i = 0; i < iters; ++i)
	{
		z = (z ^ p) + c;
		if (z.sqrMagnitude() > sqrBailout)
			break;
	}
	float k = (float)i / iters;

	// Setting point color
	buffer[offset] = (byte)(k * 255);
}

bool Mandelbrot2D::compute(size_t width, size_t height, int iters, float setScalling)
{
	sqrBailout = powf(4.0, 1.0 / (power - 1.0));
	if (setScalling < 1.0)
		setScalling = 1.0;
	width *= setScalling;
	height *= setScalling;
	if (width > fMaxFractalSize)
		width = fMaxFractalSize;
	if (height > fMaxFractalSize)
		height = fMaxFractalSize;
	this->setScalling = setScalling;
	if (points)
		delete[] points;
	this->width = width;
	this->height = height;
	int side = MAX(width, height);

	const size_t sz = side * side;
	points = new byte[sz];
	byte* dev_buffer;

	if (cudaMalloc((void**)&dev_buffer, sz) != cudaSuccess)
	{
		printf("Error on creating buffer of pixels in GPU\n");
		return false;
	}

	printf("Rendering %d^2\n", side);
	int threads = 1024;
	int blocks = (sz + threads - 1) / threads;
	clock_t tStart = clock();
	kernel << <blocks, threads >> > (dev_buffer, side, sqrBailout, power, iters);
	cudaThreadSynchronize();
	clock_t tFinish = clock();
	double tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("It tooks %.3f seconds\n", tDelta);

	printf("Moving\n");
	if (cudaMemcpy((void*)points, dev_buffer, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error on getting buffer of pixels from GPU\n");
		return false;
	}

	cudaFree(dev_buffer);
	return true;
}

void Mandelbrot2D::draw()
{
	glBegin(GL_POINTS);
	int side = MAX(width, height);
	int shiftX = (width - side) / 2 - width / 2;
	int shiftY = (height - side) / 2 - height / 2;
	for (int y = 0; y < side; ++y)
	{
		for (int x = 0; x < side; ++x)
		{
			int i = side * y + x;
			int k = points[i];
			byte kRed = colorSpectrum[k][0];
			byte kGreen = colorSpectrum[k][1];
			byte kBlue = colorSpectrum[k][2];
			glColor3ub(
				kRed,
				kGreen,
				kBlue
			);
			glVertex2f(
				(shiftX + x) / setScalling,
				(shiftY + y) / setScalling
			);
		}
	}
	glEnd();
}

void Mandelbrot2D::initColorSpectrum(int index)
{
	switch (index)
	{
	case 0:
		initColorSpectrum0();
		break;
	case 1:
		initColorSpectrum1();
		break;
	case 2:
		initColorSpectrum2();
		break;
	}
}

void Mandelbrot2D::initColorSpectrum0()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		float b = sqrtf(k);

		byte kRed = (byte)(k * 255);
		byte kGreen = (byte)(k * k * 255);
		byte kBlue = (byte)((1 - 4 * k * (1 - k)) * 255);

		colorSpectrum[i][0] = kRed * b;
		colorSpectrum[i][1] = kGreen * b;
		colorSpectrum[i][2] = kBlue * b;
	}
}

void Mandelbrot2D::initColorSpectrum1()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		float b = 4 * k * (1 - k);

		byte kRed = (byte)((k < 0.5 ? 2 * k : (k < 0.75 ? 1.0 : 4 - 4 * k)) * 255);
		byte kGreen = (byte)((k < 0.5 ? 2 * k : (k < 0.75 ? 1.5 - k : 3 - 3 * k)) * 255);
		byte kBlue = (byte)((k < 0.5 ? 1 : 2 - 2 * k) * 255);

		colorSpectrum[i][0] = kRed * b;
		colorSpectrum[i][1] = kGreen * b;
		colorSpectrum[i][2] = kBlue * b;
	}
}

void Mandelbrot2D::initColorSpectrum2()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		float b = 4 * k * (1 - k);

		byte kRed = (byte)((k < 0.5 ? 0 : (k < 0.75 ? 4 * k - 2 : 1.0)) * 255);
		byte kGreen = (byte)((k < 0.5 ? 0 : (k < 0.75 ? 4 * k - 2 : 1.0)) * 255);
		byte kBlue = (byte)((k < 0.5 ? 1 : 2 - 2 * k) * 255);

		colorSpectrum[i][0] = kRed * b;
		colorSpectrum[i][1] = kGreen * b;
		colorSpectrum[i][2] = kBlue * b;
	}
}