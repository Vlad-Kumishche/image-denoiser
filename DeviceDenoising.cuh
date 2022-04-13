#ifndef DEVICE_DENOISING_H
#define DEVICE_DENOISING_H

const int THREADS_X = 32;
const int THREADS_Y = 16;

__global__ void DeviceGrayDenoising(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight,
	const int inPitch,
	const int outPitch
);

__global__ void DeviceRgbDenoising(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight,
	const int inPitch,
	const int outPitch
);

#endif