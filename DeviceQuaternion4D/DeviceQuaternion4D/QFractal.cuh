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
	void initColorSpectrum0();
	void initColorSpectrum1();
	void initColorSpectrum2();
	void initColorSpectrum3();

public:
	int fMaxFractalSize;

	static enum ParamToHide { R, A, B, C };
	void set(float r, float a, float b, float c, QFractal::ParamToHide h);
	bool compute(size_t width, size_t height, int iters);
	void draw();
	void initColorSpectrum(int index);
};

