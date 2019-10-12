#pragma once

#ifndef __gpu_opencl_H
#define __gpu_opencl_H

#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/cl2.hpp>
#include "matrix.h"

class gpu_opencl
{
public:
	gpu_opencl() {}
	void init();
	matrix<float> matmul (matrix<float>& A, matrix<float>& B);
	~gpu_opencl() {}
private:

	std::string loadKernel(std::string fname);

	std::vector<cl::Platform> platforms;
	cl::Context context;
	std::vector<cl::Device> devices;
	cl::CommandQueue queue;
	cl::Program program;
	cl::Kernel kernel;


	std::string matmul_kernel = "matmul_kernel.ocl";
};


#endif