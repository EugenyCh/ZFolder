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
	void initColorSpectrum0();
	void initColorSpectrum1();
	void initColorSpectrum2();
	void initColorSpectrum3();

public:
	int fMaxFractalSize;
	float fPower;

	bool compute(size_t width, size_t height, int iters);
	void draw();
	void initColorSpectrum(int index);
};

