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
	float hcX = 0.0;
	float hcY = 0.0;
	float hcZ = 0.0;
	bool hasParameter = false;

public:
	int fMaxFractalSize;
	float fPower;

	void setConstParam(float x, float y, float z);
	void setNoConstParam();
	bool compute(size_t width, size_t height, int iters);
	void draw();
	void initColorSpectrum(int index);
};

