#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "QFractal.cuh"
#include "Quaternion.cuh"
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (b) : (a))

__device__ int side1;
__device__ int side2;
__device__ int side3;

__global__ void initVars(const int side)
{
	side1 = side;
	side2 = side * side;
	side3 = side * side * side;
}

__global__ void kernel(
	byte* buffer,
	const float q1,
	const float q2,
	const float q3,
	const float q4,
	const int maxIter,
	const float bailout,
	const float sqrBailout,
	int* counterPoints)
{
	int offset = threadIdx.x + blockDim.x * blockIdx.x;
	if (offset >= side3)
		return;
	int z = offset / side2;
	offset -= z * side2;
	int y = offset / side1;
	int x = offset % side1;
	offset += z * side2;

	// Compute point at this position
	int halfSide = side1 >> 1;
	float fr = bailout * (float)(x - halfSide) / halfSide;
	float fa = bailout * (float)(y - halfSide) / halfSide;
	float fb = bailout * (float)(z - halfSide) / halfSide;
	float fc = q4;
	Quaternion qc(q1, q2, q3, q4);
	Quaternion qv(fr, fa, fb, fc);

	// Iterating
	bool belongs;
	if (qv.sqrRadius() > sqrBailout)
		belongs = false;
	else
	{
		for (int i = 0; i < maxIter; ++i)
			qv = (qv ^ 5) + qc;
		belongs = qv.sqrRadius() <= sqrBailout;
	}

	if (belongs)
	{
		buffer[offset] = (byte)((fr * fr + fa * fa + fb * fb) / (sqrBailout - fc * fc) * 255);
		atomicAdd(counterPoints, 1);
	}
	else
		buffer[offset] = 0;
}

void QFractal::set(float r, float a, float b, float c, QFractal::ParamToHide h)
{
	switch (h)
	{
	case QFractal::R:
		this->q1 = a;
		this->q2 = b;
		this->q3 = c;
		this->q4 = r;
		break;
	case QFractal::A:
		this->q1 = r;
		this->q2 = b;
		this->q3 = c;
		this->q4 = a;
		break;
	case QFractal::B:
		this->q1 = r;
		this->q2 = a;
		this->q3 = c;
		this->q4 = b;
		break;
	case QFractal::C:
		this->q1 = r;
		this->q2 = a;
		this->q3 = b;
		this->q4 = c;
		break;
	}
}

bool QFractal::compute(size_t width, size_t height, int iters)
{
	if (points)
		delete[] points;
	this->width = width;
	this->height = height;
	if (width > fMaxFractalSize)
		width = fMaxFractalSize;
	if (height > fMaxFractalSize)
		height = fMaxFractalSize;
	side = MAX(width, height);

	this->bailout = 2.0f;
	this->sqrBailout = 4.0f;

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
	printf("Rendering %d^3\n", side);
	int threads = 1024;
	int blocks = (sz + threads - 1) / threads;
	int counterPoints = 0;
	int* dev_counterPoints;
	cudaMalloc((void**)&dev_counterPoints, sizeof(int));
	cudaMemcpy(dev_counterPoints, &counterPoints, sizeof(int), cudaMemcpyHostToDevice);
	// Start
	tStart = clock();
	initVars << <1, 1 >> > (side);
	kernel << <blocks, threads >> > (dev_buffer, q1, q2, q3, q4, iters, bailout, sqrBailout, dev_counterPoints);
	cudaThreadSynchronize();
	tFinish = clock();
	// End
	cudaMemcpy(&counterPoints, dev_counterPoints, sizeof(int), cudaMemcpyDeviceToHost);
	tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("Included %d points (%.1f %%)\n",
		counterPoints,
		100.f * counterPoints / sz);
	printf("It tooks %.3f seconds\n", tDelta);

	if (cudaMemcpy((void*)points, dev_buffer, sz, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error on getting buffer of pixels from GPU\n");
		return false;
	}
	cudaFree(dev_buffer);
	cudaFree(dev_counterPoints);

	printf("Cleaning of points\n");
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
				int offset = z * side * side + y * side + x;
				if (points[offset] == 0)
					continue;
				else if (points[offset] == 1)
				{
					pointsToCleaning[index++] = offset;
					++cleaned;
					continue;
				}
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
					pointsToCleaning[index++] = offset;
					++cleaned;
					continue;
				}
			}
		}
	}
	for (int i = 0; i < index; ++i)
		points[pointsToCleaning[i]] = 0;
	printf("Cleaned %d points (%.1f %%)\n",
		cleaned,
		100.f * cleaned / counterPoints);
	// End
	tFinish = clock();
	tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("It tooks %.3f seconds\n\n", tDelta);
	delete[] pointsToCleaning;
	return true;
}

void QFractal::draw()
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
						int k = points[i];
						byte kRed = colorSpectrum[k][0];
						byte kGreen = colorSpectrum[k][1];
						byte kBlue = colorSpectrum[k][2];
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

void QFractal::initColorSpectrum(int index)
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
	case 3:
		initColorSpectrum3();
		break;
	}
}

void QFractal::initColorSpectrum0()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		k = sqrtf(k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);
		float b = 1 - 3 * k * (1 - k);

		byte kRed = (byte)(4 * k * (1 - k) * 255);
		byte kGreen = (byte)(k * 127);
		byte kBlue = (byte)((1 - k) * 255);

		colorSpectrum[i][0] = kRed * b;
		colorSpectrum[i][1] = kGreen * b;
		colorSpectrum[i][2] = kBlue * b;
	}
}

void QFractal::initColorSpectrum1()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		k = sqrtf(k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);

		byte kRed = (byte)(k * 255);
		byte kGreen = (byte)(k * k * 255);
		byte kBlue = (byte)((1 - 4 * k * (1 - k)) * 255);

		colorSpectrum[i][0] = kRed;
		colorSpectrum[i][1] = kGreen;
		colorSpectrum[i][2] = kBlue;
	}
}

void QFractal::initColorSpectrum2()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		k = sqrtf(k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);
		float b = 2 * k * (1 - k) + 0.5;

		byte kRed = (byte)((k < 0.5 ? 2 * k : (k < 0.75 ? 1.0 : 3.25 - 3 * k)) * 255);
		byte kGreen = (byte)((k < 0.5 ? 2 * k : (k < 0.75 ? 1.5 - k : 2.25 - 2 * k)) * 255);
		byte kBlue = (byte)((k < 0.5 ? 1 : 2 - 2 * k) * 255);

		colorSpectrum[i][0] = kRed * b;
		colorSpectrum[i][1] = kGreen * b;
		colorSpectrum[i][2] = kBlue * b;
	}
}

void QFractal::initColorSpectrum3()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		k = sqrtf(k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);
		k = 4 * k * (1 - k);
		float b = 1 - 3 * k * (1 - k);

		byte kRed = (byte)((k < 0.5 ? 0 : (k < 0.75 ? 4 * k - 2 : 1.0)) * 255);
		byte kGreen = (byte)((k < 0.5 ? 0 : (k < 0.75 ? 2 * k - 1 : 0.5)) * 255);
		byte kBlue = (byte)((k < 0.5 ? 1 : 2 - 2 * k) * 255);

		colorSpectrum[i][0] = kRed * b;
		colorSpectrum[i][1] = kGreen * b;
		colorSpectrum[i][2] = kBlue * b;
	}
}
