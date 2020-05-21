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

public:
	Mandelbulb(float power, int maxIter);
	void compute(size_t width, size_t height);
	void draw(size_t width, size_t height);
	void initColorSpectrum();
};

