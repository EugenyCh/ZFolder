#pragma once
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef unsigned char byte;

class Julia2D
{
private:
	byte* points = nullptr;
	int width = 0;
	int height = 0;
	byte colorSpectrum[256][3];
	float setScalling;
	void initColorSpectrum0();
	void initColorSpectrum1();
	void initColorSpectrum2();
	float sqrBailout;

public:
	int fMaxFractalSize;
	float cx;
	float cy;
	float power;

	bool compute(size_t width, size_t height, int iters, float setScalling);
	void draw();
	void initColorSpectrum(int index);
};

