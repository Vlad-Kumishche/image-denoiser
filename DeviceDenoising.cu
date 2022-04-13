#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DeviceDenoising.cuh"

#include <stdio.h>

__global__ void DeviceGrayDenoising(
	unsigned char* inputData,
	unsigned char* outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight,
	const int inPitch,
	const int outPitch
)
{
	const int x = blockIdx.x * THREADS_X + threadIdx.x;
	const int y = blockIdx.y * THREADS_Y + threadIdx.y;

	const int int_widht = inPitch / sizeof(int);
	const int width_border = (w + sizeof(int) - 1) / sizeof(int);

	uchar4* thread_input = (uchar4*)(inputData);
	uchar4* thread_output = (uchar4*)(outputData);

	__shared__ uchar4 shared_memory[THREADS_Y + 2][THREADS_X + 2];

	if (x < int_widht || y < zoomedHeight)
	{
		uchar4* thread_input = (uchar4*)(inputData);
		uchar4* thread_output = (uchar4*)(outputData);

		shared_memory[threadIdx.y][threadIdx.x] = thread_input[y * int_widht + x];

		if (threadIdx.y < 2)
		{
			if (y + THREADS_Y <= h)
			{
				shared_memory[THREADS_Y + threadIdx.y][threadIdx.x] = thread_input[(THREADS_Y + y) * int_widht + x];
			}
			if (threadIdx.x < THREADS_Y + 2)
			{
				int temp_x = blockIdx.x * THREADS_X + threadIdx.y;
				int temp_y = blockIdx.y * THREADS_Y + threadIdx.x;

				if (temp_x < int_widht && temp_y < zoomedHeight)
				{
					shared_memory[threadIdx.x][THREADS_X + threadIdx.y] = thread_input[temp_y * int_widht + THREADS_X + temp_x];
				}
			}
		}
	}

	__syncthreads();

	if (y >= h || x >= width_border)
	{
		return;
	}

	uchar4 generated_int = { 0 };

	uchar4 int_1 = shared_memory[threadIdx.y][threadIdx.x];
	uchar4 int_2 = shared_memory[threadIdx.y][threadIdx.x + 1];
	uchar4 int_3 = shared_memory[threadIdx.y + 1][threadIdx.x];
	uchar4 int_4 = shared_memory[threadIdx.y + 1][threadIdx.x + 1];
	uchar4 int_5 = shared_memory[threadIdx.y + 2][threadIdx.x];
	uchar4 int_6 = shared_memory[threadIdx.y + 2][threadIdx.x + 1];

	generated_int.x = (
		int_1.x + int_1.y * 2 + int_1.z +
		int_3.x * 2 + int_3.y * 4 + int_3.z * 2 +
		int_5.x + int_5.y * 2 + int_5.z
		) / 16;

	generated_int.y = (
		int_1.y + int_1.z * 2 + int_1.w +
		int_3.y * 2 + int_3.z * 4 + int_3.w * 2 +
		int_5.y + int_5.z * 2 + int_5.w
		) / 16;

	generated_int.z = (
		int_1.z + int_1.w * 2 + int_2.x +
		int_3.z * 2 + int_3.w * 4 + int_4.x * 2 +
		int_5.z + int_5.w * 2 + int_6.x
		) / 16;

	generated_int.w = (
		int_1.w + int_2.x * 2 + int_2.y +
		int_3.w * 2 + int_4.x * 4 + int_4.y * 2 +
		int_5.w + int_6.x * 2 + int_6.y
		) / 16;

	const int output_int_width = outPitch / sizeof(int);
	thread_output[y * output_int_width + x] = generated_int;
}

__global__ void DeviceRgbDenoising(
	unsigned char *inputData,
	unsigned char *outputData,
	const int w,
	const int h,
	const int zoomedWidth,
	const int zoomedHeight,
	const int inPitch,
	const int outPitch
)
{
	const int x = blockIdx.x * THREADS_X + threadIdx.x;
	const int y = blockIdx.y * THREADS_Y + threadIdx.y;

	const int int_widht = inPitch / sizeof(int);
	const int width_border = (w + sizeof(int) - 1) / sizeof(int);

	uchar4 *thread_input = (uchar4*)(inputData);
	uchar4 *thread_output = (uchar4*)(outputData);

	__shared__ uchar4 shared_memory[THREADS_Y + 2][THREADS_X + 2];

	if (x < int_widht && y < h)
	{
		uchar4 *thread_input = (uchar4*)(inputData);
		uchar4 *thread_output = (uchar4*)(outputData);

		shared_memory[threadIdx.y][threadIdx.x] = thread_input[y * int_widht + x];

		if (threadIdx.y < 2)
		{
			if (y + THREADS_Y < zoomedHeight)
			{
				shared_memory[THREADS_Y + threadIdx.y][threadIdx.x] = thread_input[(THREADS_Y + y) * int_widht + x];
			}
			if (threadIdx.x < THREADS_Y + 2)
			{
				int temp_x = blockIdx.x * THREADS_X + threadIdx.y;
				int temp_y = blockIdx.y * THREADS_Y + threadIdx.x;

				if (temp_x < int_widht && temp_y < zoomedHeight)
				{
					shared_memory[threadIdx.x][THREADS_X + threadIdx.y] = thread_input[temp_y * int_widht + THREADS_X + temp_x];
				}
			}
		}
	}


	__syncthreads();

	if (x < width_border && y < h)
	{
		uchar4 generated_int = { 0 };

		uchar4 int_1 = shared_memory[threadIdx.y][threadIdx.x];
		uchar4 int_2 = shared_memory[threadIdx.y][threadIdx.x + 1];
		uchar4 int_3 = shared_memory[threadIdx.y][threadIdx.x + 2];
		uchar4 int_4 = shared_memory[threadIdx.y + 1][threadIdx.x];
		uchar4 int_5 = shared_memory[threadIdx.y + 1][threadIdx.x + 1];
		uchar4 int_6 = shared_memory[threadIdx.y + 1][threadIdx.x + 2];
		uchar4 int_7 = shared_memory[threadIdx.y + 2][threadIdx.x];
		uchar4 int_8 = shared_memory[threadIdx.y + 2][threadIdx.x + 1];
		uchar4 int_9 = shared_memory[threadIdx.y + 2][threadIdx.x + 2];

		generated_int.x =
		(
			int_1.x + int_1.w * 2 + int_2.z +
			int_4.x * 2 + int_4.w * 4 + int_5.z * 2 +
			int_7.x + int_7.w * 2 + int_8.z
		) / 16;

		generated_int.y =
		(
			int_1.y + int_2.x * 2 + int_2.w +
			int_4.y * 2 + int_5.x * 4 + int_5.w * 2 +
			int_7.y + int_8.x * 2 + int_8.w
		) / 16;

		generated_int.z =
		(
			int_1.z + int_2.y * 2+ int_3.x +
			int_4.z * 2 + int_5.y * 4 + int_6.x * 2 +
			int_7.z + int_8.y * 2+ int_9.x
		) / 16;

		generated_int.w =
		(
			int_1.w + int_2.z * 2+ int_3.y +
			int_4.w * 2 + int_5.z * 4 + int_6.y * 2 +
			int_7.w + int_8.z * 2+ int_9.y
		) / 16;

		const int output_int_width = outPitch / sizeof(int);
		thread_output[y * output_int_width + x] = generated_int;
	}
}