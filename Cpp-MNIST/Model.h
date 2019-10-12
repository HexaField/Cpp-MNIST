#pragma once
#include <iostream>
#include "DenseLayer.h"
#include "matrix.h"
#include "gpu_opencl.h"

class Model
{
public:
	Model(bool use, gpu_opencl g) { use_gpu = use; gpu = g; }
	Model() { use_gpu = false; }
	~Model();

	void addLayer(DenseLayer layer);
	void generateWeights(int input_size);
	void print_matrix(matrix<float> mat);
	matrix<float> feed_forward(matrix<float> input);
	void back_propagate(matrix<float> error_delta, matrix<float> input, float learning_rate);


	float relu(float x);
	matrix<float> relu(matrix<float> mat);
	float relu_prime(float x);
	matrix<float> relu_prime(matrix<float> mat);
	matrix<float> softmax(matrix<float> mat);
	matrix<float> dot(matrix<float> A, matrix<float> B);

	void debug()
	{
		print_matrix(weights[0]);
		print_matrix(bias[0]);
		print_matrix(layer_error[0]);
	}
	matrix<float> randomise(matrix<float> mat);
private:
	bool use_gpu;
	gpu_opencl gpu;
	std::vector<DenseLayer> layers;
	std::vector<matrix<float>> weights;
	std::vector<matrix<float>> bias;
	std::vector<matrix<float>> outputs;
	std::vector<matrix<float>> layer_error;
	std::vector<matrix<float>> layer_error_deriv;
	std::vector<matrix<float>> layer_weight_delta;

};

