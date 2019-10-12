#include "pch.h"
#include "Model.h"
#include <vector>


matrix<float> Model::dot(matrix<float> A, matrix<float> B)
{
	if (use_gpu)
	{
		if (A.get_cols() != B.get_rows())
		{
			std::cout << "Matricies must align: " << A.get_rows() << ", " << A.get_cols() << " x " << B.get_rows() << ", " << B.get_cols() << std::endl;
			exit(-1);
		}

		return gpu.matmul(A, B);
	}
	else
	{
		return A.dot(B);
	}
}

float Model::relu(float x)
{
	return x >= 0 ? x : 0;
}

matrix<float> Model::relu(matrix<float> mat)
{
	matrix<float> output(mat);
	for (unsigned i = 0; i < mat.get_rows(); i++)
	{
		for (unsigned j = 0; j < mat.get_cols(); j++)
		{
			output(i, j) = relu(mat(i, j));
		}
	}
	return output;
}

float Model::relu_prime(float x)
{
	return x >= 0 ? 1 : 0;
}

matrix<float> Model::relu_prime(matrix<float> mat)
{
	matrix<float> output(mat);
	for (unsigned i = 0; i < mat.get_rows(); i++)
	{
		for (unsigned j = 0; j < mat.get_cols(); j++)
		{
			output(i, j) = relu_prime(mat(i, j));
		}
	}
	return output;
}

matrix<float> Model::softmax(matrix<float> mat)
{
	std::vector<std::vector<float>> output;
	const unsigned num_outputs = mat.get_cols();
	for (unsigned i = 0; i < mat.get_rows(); i++)
	{
		std::vector<float> values;
		std::vector<float> output_values;
		for (unsigned j = 0; j < num_outputs; j++)
		{
			values.push_back(mat(i, j));
		}

		float max_value = *std::max_element(values.begin(), values.end());

		for (unsigned j = 0; j < num_outputs; j++)
		{
			values[j] = exp(values[j] - max_value);
		}

		float sum_of_elems = 0.0f;
		for (unsigned j = 0; j < num_outputs; j++)
		{
			sum_of_elems += values[j];
		}

		for (unsigned j = 0; j < num_outputs; j++)
		{
			output_values.push_back(values[j] / sum_of_elems);
		}
		output.push_back(output_values);
	}
	return matrix<float> (output);
}

void Model::print_matrix(matrix<float> mat)
{
	for (unsigned i = 0; i < mat.get_rows(); i++)
	{
		for (unsigned j = 0; j < mat.get_cols(); j++)
		{
			std::cout << mat(i, j) << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void Model::addLayer(DenseLayer layer)
{
	layers.push_back(layer);
}

matrix<float> Model::randomise(matrix<float> mat)
{
	matrix<float> output(mat);
	for (unsigned i = 0; i < mat.get_rows(); i++)
	{
		for (unsigned j = 0; j < mat.get_cols(); j++)
		{
			output(i, j) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}
	return output;
}

void Model::generateWeights(int input_size)
{
	std::cout << std::endl << "Network shape: " << input_size;
	for (int i = 0; i < layers.size(); i++)
	{

		weights.push_back(matrix<float>(i == 0 ? input_size : layers[i - 1].get_num_neurons(), layers[i].get_num_neurons(), 0.0f));
		weights[i] = randomise(weights[i]);
		bias.push_back(matrix<float>(1, layers[i].get_num_neurons(), 0.0f));
		std::cout << " -> " << layers[i].get_num_neurons();
	}
	std::cout << std::endl << std::endl;
}

// forward pass
matrix<float> Model::feed_forward(matrix<float> input)
{
	outputs.clear();
	matrix<float> layer_input(input);
	for (int i = 0; i < layers.size(); i++)
	{
		matrix<float> activation(dot(layer_input, weights[i]));
		activation += bias[i];
		if (i == layers.size() - 1)
			layer_input = softmax(activation);
		else
			layer_input = relu(activation);
		//print_matrix(layer_input);
		outputs.push_back(layer_input);
	}
	return layer_input;
}

void Model::back_propagate(matrix<float> error, matrix<float> input, float learning_rate)
{
	layer_error.clear();
	layer_error_deriv.clear();
	layer_weight_delta.clear();

	layer_error.push_back(error);
	if (layers.size() > 1)
		layer_error_deriv.push_back(error * relu_prime(outputs[layers.size() - 1]));
	else
		layer_error_deriv.push_back(error);
	layer_weight_delta.push_back(dot(layers.size() <= 1 ? input.transpose() : outputs[layers.size() - 2].transpose(), layer_error_deriv[0]));

	if (layers.size() > 1)
	{
		for (int i = layers.size() - 2; i > 0; i--) 
		{
			layer_error.insert(layer_error.begin(), dot(layer_error_deriv[0], weights[i + 1].transpose()));
			layer_error_deriv.insert(layer_error_deriv.begin(), layer_error[0] * relu_prime(outputs[i]));
			layer_weight_delta.insert(layer_weight_delta.begin(), dot(outputs[i - 1].transpose(), layer_error_deriv[0]));
		}
		layer_error.insert(layer_error.begin(), dot(layer_error_deriv[0], weights[1].transpose()));
		layer_error_deriv.insert(layer_error_deriv.begin(), layer_error[0] * relu_prime(outputs[0]));
		layer_weight_delta.insert(layer_weight_delta.begin(), dot(input.transpose(), layer_error_deriv[0]));
	}

	for (int i = 0; i < layers.size(); i++)
	{
		//print_matrix(layer_weight_delta[i]);
		weights[i] -= layer_weight_delta[i] * learning_rate;
		bias[i] -= layer_error_deriv[i].sum_v() * learning_rate;
	}
}

Model::~Model()
{
}
