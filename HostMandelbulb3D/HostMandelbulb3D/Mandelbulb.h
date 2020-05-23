#pragma once
typedef unsigned char byte;

class Mandelbulb
{
private:
	float n;
	int maxIter;
	float bailout;
	float sqrBailout;
	byte* points = nullptr;
	size_t side = 0;
	byte colorSpectrum[256][3];
	int width = 0;
	int height = 0;

public:
	Mandelbulb(float power, int maxIter);
	void compute(size_t width, size_t height);
	void draw();
	void initColorSpectrum();
};

