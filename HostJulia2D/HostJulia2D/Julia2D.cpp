#include "Julia2D.h"
#include "Complex.h"
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>
#include <stdio.h>
#include <time.h>

Julia2D::Julia2D(float cx, float cy)
{
	this->cx = cx;
	this->cy = cy;
}

bool Julia2D::compute(size_t width, size_t height, int iters, float setScalling)
{
	if (setScalling < 1.0)
		setScalling = 1.0;
	width *= setScalling;
	height *= setScalling;
	this->setScalling = setScalling;
	if (points)
		delete[] points;
	this->width = width;
	this->height = height;
	int side = MAX(width, height);

	const size_t sz = side * side;
	points = new byte[sz];

	printf("Rendering %d^2\n", side);
	int halfSide = side >> 1;
	clock_t tStart = clock();
	// Processing
	for (int y = 0; y < side; ++y)
	{
		for (int x = 0; x < side; ++x)
		{
			int offset = y * side + x;
			// Compute point at this position
			float jx = 2.0f * (float)(x - halfSide) / halfSide;
			float jy = 2.0f * (float)(y - halfSide) / halfSide;
			Complex c(cx, cy);
			Complex z(jx, jy);

			// Iterating
			int i;
			for (i = 0; i < iters; ++i)
			{
				z = z * z + c;
				if (z.sqrMagnitude() > 4.0f)
					break;
			}
			float k = (float)i / iters;

			// Setting point color
			points[offset] = (byte)(k * 255);
		}
	}
	// End
	clock_t tFinish = clock();
	double tDelta = (double)(tFinish - tStart) / CLOCKS_PER_SEC;
	printf("It tooks %.3f seconds\n", tDelta);

	return true;
}

void Julia2D::draw()
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
			glColor3ub(k, k, k);
			glVertex2f(
				(shiftX + x) / setScalling,
				(shiftY + y) / setScalling
			);
		}
	}
	glEnd();
}