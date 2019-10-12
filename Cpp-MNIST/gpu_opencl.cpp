#include "pch.h"
#include "gpu_opencl.h"

#ifndef __gpu_opencl_CPP
#define __gpu_opencl_CPP

#include <string>
#include <iostream>
#include <fstream>
#include <streambuf>
#include <iterator>

std::string gpu_opencl::loadKernel(std::string fname)
{
	std::ifstream t(fname);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}

void gpu_opencl::init()
{
	try
	{
		std::cout << "Initialising GPU..." << std::endl;
		cl::Platform::get(&platforms);
		std::vector<cl::Platform>::iterator iter;
		//for (iter = platforms.begin(); iter != platforms.end(); ++iter)
		//{
			//if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc."))
			//{
			//}
		//}

		cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, reinterpret_cast <cl_context_properties>(platforms[0]()), 0 };
		context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
		devices = context.getInfo<CL_CONTEXT_DEVICES>();

		std::cout << std::endl << "Using device " + devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
		std::cout << " - CL_DEVICE_VERSION: " << devices[0].getInfo<CL_DEVICE_VERSION>() << std::endl;
		std::cout << " - CL_DRIVER_VERSION: " << devices[0].getInfo<CL_DRIVER_VERSION>() << std::endl;
		std::cout << " - CL_DEVICE_ADDRESS_BITS: " << devices[0].getInfo<CL_DEVICE_ADDRESS_BITS>() << std::endl;
		std::cout << " - CL_DEVICE_GLOBAL_MEM_SIZE: " << devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
		std::cout << " - CL_DEVICE_LOCAL_MEM_SIZE: " << devices[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
		std::cout << " - CL_DEVICE_MAX_WORK_GROUP_SIZE: " << devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
		std::cout << " - CL_DEVICE_MAX_COMPUTE_UNITS: " << devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl << std::endl;

		queue = cl::CommandQueue(context, devices[0]);

		std::string matmul_kernel_source = loadKernel("matmul_kernel2.ocl");
		cl::Program::Sources sources(1, std::make_pair(matmul_kernel_source.c_str(), matmul_kernel_source.length()));
		program = cl::Program(context, sources);
		program.build(devices);
		kernel = cl::Kernel(program, "matrixMultiplication");
		
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
		if (err.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			std::cout << "Program Info: " << str << std::endl;
		}
	}
	catch (std::string msg)
	{
		std::cerr << "Exception caught in matmul(): " << msg << std::endl;
	}
}

matrix<float> gpu_opencl::matmul(matrix<float>& A_matrix, matrix<float>& B_matrix)
{
	if (A_matrix.get_cols() != B_matrix.get_rows())
	{
		std::cout << "ERROR: Matricies don't align! Got " << A_matrix.get_rows() << ", " << A_matrix.get_cols() << " x " << B_matrix.get_rows() << ", " << B_matrix.get_cols() << std::endl;
		exit(-3);
	}

	cl_int M = A_matrix.get_rows();
	cl_int N = B_matrix.get_cols();
	cl_int K = A_matrix.get_cols();

	cl_int* M_pointer;
	cl_int* N_pointer;
	cl_int* K_pointer;

	M_pointer = &M;
	N_pointer = &N;
	K_pointer = &K;

	std::vector<float> A_vec = A_matrix.flatten();
	std::vector<float> B_vec = B_matrix.flatten();
	std::vector<float> C_vec(M * N, 0);

	try
	{
		cl_int error[3];
		cl::Buffer buffer_M(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &M_pointer, &error[0]);
		cl::Buffer buffer_N(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &N_pointer, &error[1]);
		cl::Buffer buffer_K(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &K_pointer, &error[2]);
		if (error[0] != CL_SUCCESS)
		{
			std::cerr << "Error: cl::Buffer failed1. (" << error[0] << ") ";
			exit(1);
		}

		queue.enqueueWriteBuffer(buffer_M, CL_TRUE, 0, sizeof(cl_int), &M);
		queue.enqueueWriteBuffer(buffer_N, CL_TRUE, 0, sizeof(cl_int), &N);
		queue.enqueueWriteBuffer(buffer_K, CL_TRUE, 0, sizeof(cl_int), &K);

		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * A_vec.size(), &A_vec.at(0), &error[0]);
		cl::Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * B_vec.size(), &B_vec.at(0), &error[1]);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * C_vec.size(), &C_vec, &error[2]);

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(cl_float) * A_vec.size(), &A_vec.at(0));
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(cl_float) * B_vec.size(), &B_vec.at(0));

		const int TS = 32;
		const int WPT = 8;
		const int RTS = 4;

		kernel.setArg(0, sizeof(cl_int), (void*)&buffer_M);
		kernel.setArg(1, sizeof(cl_int), (void*)&buffer_N);
		kernel.setArg(2, sizeof(cl_int), (void*)&buffer_K);
		kernel.setArg(3, sizeof(cl_mem), (void*)&buffer_A);
		kernel.setArg(4, sizeof(cl_mem), (void*)&buffer_B);
		kernel.setArg(5, sizeof(cl_mem), (void*)&buffer_C);
		
		//std::cout << "type " << buffer_C.getInfo<CL_MEM_TYPE>() << std::endl;
		//std::cout << "size " << buffer_C.getInfo< CL_MEM_SIZE>() << std::endl;
		//std::cout << "pointer " << buffer_C.getInfo< CL_MEM_HOST_PTR>() << std::endl;
		//std::cout << "flags " << buffer_C.getInfo< CL_MEM_FLAGS>() << std::endl;
		std::cout << M << " x " << N << std::endl;

		queue.enqueueNDRangeKernel(kernel, NULL, cl::NDRange(M, N / WPT), cl::NDRange(TS, RTS), NULL, NULL);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(cl_float) * C_vec.size(), &C_vec);

	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
		exit(-10);
	}
	catch (std::string msg)
	{
		std::cerr << "Exception caught in matmul(): " << msg << std::endl;
	}
	//std::cout << "Success!";
	matrix<float> result(M, N, C_vec);
	return result;
}

#endif