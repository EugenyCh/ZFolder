#pragma once
#include "Quaternion.cuh"
typedef unsigned char byte;

class QFractal
{
private:
	float q1, q2, q3, q4;
	int maxIter;
	float bailout;
	float sqrBailout;
	byte* points = nullptr;
	size_t side = 0;
	byte colorSpectrum[256][3];
	int width = 0;
	int height = 0;

public:
	static enum ParamToHide { R, A, B, C };
	QFractal(float r, float a, float b, float c, ParamToHide h, int maxIter);
	bool compute(size_t width, size_t height);
	void draw();
	void initColorSpectrum();
};

