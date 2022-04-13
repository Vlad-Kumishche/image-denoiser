#include <iostream>
#include <iomanip>

#include "HostDenoising.h"

using namespace std;

unsigned char* ZoomBorders (
	unsigned char* inputData,
	int w,
	int h
)
{
	const int new_width = w + 2;
	const int new_height = h + 2;

	unsigned char* output_data = new unsigned char[new_width * new_height];

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			output_data[(y + 1) * new_width + x + 1] = inputData[y * w + x];
		}
	}

	output_data[0] = inputData[0];
	output_data[new_width - 1] = inputData[w - 1];
	output_data[new_width * (new_height - 1)] = inputData[w * (h - 1)];
	output_data[new_width * new_height - 1] = inputData[w * h - 1];

	for (int x = 0; x < w; x++)
	{
		output_data[x + 1] = inputData[x];
		output_data[(new_height - 1) * new_width + x + 1] = inputData[w * (h - 1) + x];
	}

	for (int y = 0; y < h; y++)
	{
		output_data[(y + 1) * new_width] = inputData[y * w];
		output_data[(y + 1) * new_width + new_width - 1] = inputData[y * w + w - 1];
	}

	return output_data;
}

unchar3* ZoomBorders (
	unchar3 *inputData,
	int w,
	int h
)
{
	const int new_width = w + 2;
	const int new_height = h + 2;

	unchar3 *output_data = new unchar3[new_width * new_height];

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			output_data[(y + 1) * new_width + x + 1] = inputData[y * w + x];
		}
	}

	output_data[0] = inputData[0];
	output_data[new_width - 1] = inputData[w - 1];
	output_data[new_width * (new_height - 1)] = inputData[w * (h - 1)];
	output_data[new_width * new_height - 1] = inputData[w * h - 1];

	for (int x = 0; x < w; x++)
	{
		output_data[x + 1] = inputData[x];
		output_data[(new_height - 1) * new_width + x + 1] = inputData[w * (h - 1) + x];
	}

	for (int y = 0; y < h; y++)
	{
		output_data[(y + 1) * new_width] = inputData[y * w];
		output_data[(y + 1) * new_width + new_width - 1] = inputData[y * w + w - 1];
	}

	return output_data;
}

void GrayFilter(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight
)
{
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			// top row
			unsigned char t1 = inputData[y * zoomedWidth + x];
			unsigned char t2 = inputData[y * zoomedWidth + x + 1];
			unsigned char t3 = inputData[y * zoomedWidth + x + 2];

			// middle row
			unsigned char m1 = inputData[(y + 1) * zoomedWidth + x];
			unsigned char m2 = inputData[(y + 1) * zoomedWidth + x + 1];
			unsigned char m3 = inputData[(y + 1) * zoomedWidth + x + 2];

			//bottom row
			unsigned char b1 = inputData[(y + 2) * zoomedWidth + x];
			unsigned char b2 = inputData[(y + 2) * zoomedWidth + x + 1];
			unsigned char b3 = inputData[(y + 2) * zoomedWidth + x + 2];

			unsigned char result = (
				// top row
				inputData[y * zoomedWidth + x] +
				inputData[y * zoomedWidth + x + 1] * 2 +
				inputData[y * zoomedWidth + x + 2] +

				// middle row
				inputData[(y + 1) * zoomedWidth + x] * 2 +
				inputData[(y + 1) * zoomedWidth + x + 1] * 4 +
				inputData[(y + 1) * zoomedWidth + x + 2] * 2 +

				//bottom row
				inputData[(y + 2) * zoomedWidth + x] +
				inputData[(y + 2) * zoomedWidth + x + 1] * 2 +
				inputData[(y + 2) * zoomedWidth + x + 2]
				) / 16;

			outputData[y * w + x] = result;

		}
	}
}


void HostGrayDenoising(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h
)
{
	unsigned char* zoomedData = ZoomBorders(inputData, w, h);

	const int zoomedWidth = w + 2;
	const int zoomedHeight = h + 2;

	GrayFilter(zoomedData, outputData, w, h, zoomedWidth, zoomedHeight);
}

void RgbFilter(
	unchar3 *inputData,
	unchar3 *outputData,
	const int w, 
	const int h,
	const int zoomedWidth,
	const int zoomedHeight
)
{
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			unchar3 result = { 0 };

			// top row
			unchar3 t1 = inputData[y * zoomedWidth + x];
			unchar3 t2 = inputData[y * zoomedWidth + x + 1];
			unchar3 t3 = inputData[y * zoomedWidth + x + 2];

			// middle row
			unchar3 m1 = inputData[(y + 1) * zoomedWidth + x];
			unchar3 m2 = inputData[(y + 1) * zoomedWidth + x + 1];
			unchar3 m3 = inputData[(y + 1) * zoomedWidth + x + 2];

			// bottom row
			unchar3 b1 = inputData[(y + 2) * zoomedWidth + x];
			unchar3 b2 = inputData[(y + 2) * zoomedWidth + x + 1];
			unchar3 b3 = inputData[(y + 2) * zoomedWidth + x + 2];

			result.r =
			( 
				t1.r + t2.r * 2 + t3.r +
				m1.r * 2 + m2.r * 4 + m3.r * 2 +
				b1.r + b2.r * 2 + b3.r
			) / 16;

			result.g = 
			(
				t1.g + t2.g * 2 + t3.g +
				m1.g * 2 + m2.g * 4 + m3.g * 2 +
				b1.g + b2.g * 2 + b3.g
			) / 16;

			result.b =
			(
				t1.b + t2.b * 2 + t3.b +
				m1.b * 2 + m2.b * 4 + m3.b * 2 +
				b1.b + b2.b * 2 + b3.b
			) / 16;

			outputData[y * w + x] = result;
			
		}
	}
}

void HostRgbDenoising(
	unchar3 *inputData,
	unchar3 *outputData,
	const int w,
	const int h
)
{
	unchar3 *zoomedData = ZoomBorders(inputData, w, h);

	const int zoomedWidth = w + 2;
	const int zoomedHeight = h + 2;

	RgbFilter(zoomedData, outputData, w, h, zoomedWidth, zoomedHeight);
}