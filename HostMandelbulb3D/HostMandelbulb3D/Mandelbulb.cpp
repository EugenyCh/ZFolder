#include "Mandelbulb.h"
#include "Hypercomplex.h"
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (b) : (a))
#define SIDE_MAX 600

Mandelbulb::Mandelbulb(float power, int maxIter)
{
	this->n = power;
	this->maxIter = maxIter;
	this->bailout = powf(2.0f, 1.0f / (power - 1));
	this->sqrBailout = powf(4.0f, 1.0f / (power - 1));
}

void Mandelbulb::compute(size_t width, size_t height)
{
	if (points)
		delete[] points;
	side = MIN(width, height);
	if (side > SIDE_MAX)
		side = SIDE_MAX;

	const size_t sz = side * side * side;
	points = new byte[sz];
	clock_t tStart, tFinish;
	double tDelta;

	printf("Rendering %d^3 points\n", side);
	int pointsCount = 0;
	int halfSide = side >> 1;
	// Processing
	tStart = clock();
	for (int z = 0; z < side; ++z)
	{
		for (int y = 0; y < side; ++y)
		{
			for (int x = 0; x < side; ++x)
			{
				int offset = z * side * side + y * side + x;
				// Compute point at this position
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
				{
					points[offset] = (byte)((hc.sqrRadius() / sqrBailout) * 255);
					++pointsCount;
				}
				else
					points[offset] = 0;
			}
		}
		printf("\r%5.1f %% | %d points (%.1f %%)",
			100.0f * z / (side - 1),
			pointsCount,
			100.f * pointsCount / sz);
	}
	tFinish = clock();
	// End
	tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("\nIncluded %d points (%.1f %%)\n",
		pointsCount,
		100.f * pointsCount / sz);
	printf("It tooks %.3f seconds\n", tDelta);
	// Cleaning
	printf("Cleaning of points\n");
	int* pointsToCleaning = new int[pointsCount];
	int cleaned = 0;
	int index = 0;
	tStart = clock();
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
				}
			}
		}
		printf("\r%5.1f %% | cleaned %d points (%.1f %%)",
			100.0f * (z - 1) / (side - 3),
			cleaned,
			100.f * cleaned / pointsCount);
	}
	for (int i = 0; i < index; ++i)
		points[pointsToCleaning[i]] = 0;
	// End
	tFinish = clock();
	tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("\nIt tooks %.3f seconds\n\n", tDelta);
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

void Mandelbulb::initColorSpectrum()
{
	for (int i = 0; i < 256; ++i)
	{
		float k = i / 255.0;
		k = sqrtf(k);
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