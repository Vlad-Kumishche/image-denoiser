#ifndef HOST_DENOISING_H
#define HOST_DENOISING_H

typedef struct unchar3
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

unsigned char* ZoomBorders (
	unsigned char* inputData,
	int w,
	int h
);

unchar3* ZoomBorders (
	unchar3* inputData,
	int w,
	int h
);

void HostGrayDenoising(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h
);

void HostRgbDenoising(
	unchar3* inputData,
	unchar3* outputData,
	const int w,
	const int h
);

#endif