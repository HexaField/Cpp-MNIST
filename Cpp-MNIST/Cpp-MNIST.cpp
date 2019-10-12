#include "pch.h"
#include <math.h>
#include <Random>
#include <vector>
#include <iostream>
#include <cmath>
#include <ctime>
#include "mnist/mnist_reader.hpp"
#include "Model.h"
#include "gpu_opencl.h"

const char* MNIST_DATA_LOCATION = "E:/Programming/datasets/mnist";


//std::vector<float> dot_cpu(std::vector<float> inputA, std::vector<float> inputB) {}
void print_matrix(matrix<float> mat)
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

void print(const char* str)
{
	std::cout << str << std::endl;
}

void print(int str)
{
	std::cout << str << std::endl;
}

float square_error(matrix<float> prediction, matrix<float> output, int batch_size)
{
	float result = 0.0f;
	result += (prediction - output).sum() * (prediction - output).sum();
	return result / batch_size;
}

float get_num_correct(matrix<float> prediction, matrix<float> output)
{
	unsigned correct = 0;
	//print_matrix(prediction);
	//print_matrix(output);
	//std::cout << prediction.get_size_str() << std::endl;
	for (unsigned i = 0; i < prediction.get_rows(); i++)
	{
		float max_output = 0;
		unsigned max_label = 0;
		unsigned correct_label = 0;
		for (unsigned j = 0; j < prediction.get_cols(); j++)
		{
			if (prediction(i, j) > max_output)
			{
				max_output = prediction(i, j);
				max_label = j;
			}
			if (output(i, j) > 0)
			{
				correct_label = j;
			}
		}
		//std::cout << correct_label << " - " << max_label << std::endl;
		if (correct_label == max_label)
		{
			correct++;
		}
	}
	return correct;
}

matrix<float> convertToFloatMatrix(matrix<uint8_t> originalMatrix)
{
	int rows = originalMatrix.get_rows();
	int cols = originalMatrix.get_cols();

	matrix<float> output(rows, cols, 0.f);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			output(i, j) = (float) originalMatrix(i, j);
		}
	}

	return output;
}

matrix<float> eye(std::vector<float>& rhs)
{
	unsigned rows = rhs.size();
	unsigned max_label = 0;
	for (unsigned i = 0; i < rows; i++)
	{
		if (rhs[i] > max_label)
			max_label = rhs[i];
	}
	max_label++;

	matrix<float> result(rows, max_label, 0.0f);

	for (unsigned i = 0; i < rows; i++)
	{
		result(i, rhs[i]) = 1.0f;
	}

	return result;
}


// only suitable for square images!
// will shrink the image, as this is not padded
// TODO: add stride
matrix<float> convolution(matrix<float> m, std::vector<std::vector<float>> kernel)
{
	unsigned kernel_size = kernel.size();
	unsigned num_images = m.get_rows();
	unsigned image_size = sqrt(m.get_cols());
	unsigned new_image_size = image_size - kernel_size + 1;

	std::cout << "Applying convolution layer with size " << kernel_size << ". (" << image_size << " -> " << new_image_size << ")" << std::endl;

	matrix<float> result(num_images, pow(new_image_size, 2), 0.0f);

	unsigned kernel_centre = round(kernel_size / 2);

	for (unsigned image_index = 0; image_index < num_images; image_index++)
	{
		for (unsigned pixel_x = kernel_centre; pixel_x < image_size - kernel_centre; pixel_x++)
		{
			for (unsigned pixel_y = kernel_centre; pixel_y < image_size - kernel_centre; pixel_y++)
			{
				for (unsigned kernel_x = 0; kernel_x < kernel_size; kernel_x++)
				{
					for (unsigned kernel_y = 0; kernel_y < kernel_size; kernel_y++)
					{
						float k_x = pixel_x + (kernel_centre - kernel_x);
						float k_y = pixel_y + (kernel_centre - kernel_y);
						result(pixel_x, pixel_y) += m(k_x, k_y) * kernel[kernel_x][kernel_y];
					}
				}
			}
		}
	}
	return result;
}

matrix<float> max_pool(matrix<float> m, unsigned pool_size)
{
	unsigned num_images = m.get_rows();
	unsigned image_size = sqrt(m.get_cols());
	unsigned new_image_size = image_size / pool_size;
	std::cout << "Applying Max Pool layer with size " << pool_size << ". (" << image_size << " -> "<< new_image_size << ")"<< std::endl;

	matrix<float> result(num_images, pow(new_image_size, 2), 0.0f);

	for (unsigned image_index = 0; image_index < num_images; image_index++)
	{
		for (unsigned pixel_x = 0; pixel_x < new_image_size; pixel_x++)
		{
			for (unsigned pixel_y = 0; pixel_y < new_image_size; pixel_y++)
			{
				float max = 0;
				for (unsigned kernel_x = 0; kernel_x < pool_size; kernel_x++)
				{
					for (unsigned kernel_y = 0; kernel_y < pool_size; kernel_y++)
					{
						float current = m(pixel_x * 2 + kernel_x, pixel_y * 2 + kernel_y);
						if (current > max)
							max = current;
					}
				}
				result(pixel_x, pixel_y) = max;
			}
		}
	}
	return result;
}

matrix<float> convolution_feature_maps_max_pooled(matrix<float> m, bool include_original)
{
	std::vector<std::vector<std::vector<float>>> kernels = {

		// vert & horiz

		{{4, 2, 1, 2, 4},
		{2, 1, 0, 1, 2},
		{1, 0, 0, 0, 1},
		{2, 1, 0, 1, 2},
		{4, 2, 1, 2, 4}},

		{{-2, -1, 0, 1, 2},
		{-2, -1, 0, 1, 2},
		{-4, -2, 0, 2, 4},
		{-2, -1, 0, 1, 2},
		{-2, -1, 0, 1, 2}},

		{{-2, -2, -4,-2, -2},
		{-1, -1, -2, -1, -1},
		{0, 0, 0, 0, 0},
		{1, 1, 2, 1, 1},
		{2, 2, 4, 2, 2}},

		{{2, 1, 0,-1, -2},
		{2, 1, 0, -1, -2},
		{4, 2, 0, -2, -4},
		{2, 1, 0, -1, -2},
		{2, 1, 0, -1, -2}},

		{{2, 2, 4, 2, 2},
		{1, 1, 2, 1, 1},
		{0, 0, 0, 0, 0},
		{-1, -1, -2, -1, -1},
		{-2, -2, -4, -2, -2}}/*,

		// diagonals

		{{0, 0, -1,-2, -4},
		{0, 0, 0, -1, -2},
		{1, 0, 0, 0, -1},
		{2, 1, 0, 0, 0},
		{4, 2, 1, 0, 0},},


		{{0, -1, -2},
		{1, 0, -1},
		{2, 1, 0}},

		{{2, 1, 0},
		{1, 0, -1},
		{0, -1, -2}},

		{{0, 1, 2},
		{1, 0, 1},
		{-2, -1, 0}}*/
	};

	unsigned kernel_size = kernels[0].size();
	unsigned num_images = m.get_rows();
	unsigned image_size = sqrt(m.get_cols());
	unsigned new_image_size = image_size - kernel_size + 1;
	matrix<float> result(num_images, 1, 0); // just to initialise

	std::cout << std::endl << "Creating " << kernels.size() << " feature maps..." << std::endl;
	for (unsigned kernel = 0; kernel < kernels.size(); kernel++)
	{
		std::cout << " - ";
		matrix<float> feature_map = convolution(m, kernels[kernel]);
		feature_map /= 4;
		std::cout << " - ";
		feature_map = max_pool(feature_map, 2);
		if (kernel == 0)
		{
			result = feature_map;
		}
		else
		{
			result = result.concatenate_v(feature_map);
		}

		std::cout << "Created feature map " << (kernel + 1) << " of size " << sqrt(feature_map.get_cols()) << " x " << sqrt(feature_map.get_cols()) << "." << std::endl;
	}
	if (include_original)
		result = result.concatenate_v(m);
	std::cout << std::endl;
	return result;
}

int main(int argc, char *argv[])
{
	//gpu_opencl gpu;
	//gpu.init();

	print("Attempting dataset load...");
	mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
		mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

	print("Processing dataset...");

	matrix<uint8_t> inputs_raw(dataset.training_images);
	clock_t time;
	time = clock();

	matrix<float> inputs = convertToFloatMatrix(inputs_raw);
	std::vector<std::vector<float>> kernel = {
		{1, 0, 1},
		{0, 1, 0},
		{1, 0, 1}
	};

	//inputs = convolution(inputs, kernel);
	//inputs = max_pool(inputs, 2);

	//inputs = convolution_feature_maps_max_pooled(inputs, false);
	//inputs = convolution_feature_maps_max_pooled(inputs, false);

	float factor = 0.98f / 255.0f; 
	inputs *= factor;
	inputs += 0.01f;

	time = clock() - time;
	std::cout << "Successfully convolved, pooled, generate feature maps and scaled inputs. Took " << ((float)time / CLOCKS_PER_SEC) << " seconds." << std::endl;

	print("Converted and scaled inputs successfully.");
	std::cout << "Inputs size: " << inputs.get_size_str() << std::endl;

	std::vector<uint8_t> outputs_raw(dataset.training_labels);
	std::vector<float> outputs_raw_float(outputs_raw.begin(), outputs_raw.end());
	matrix<float> outputs = eye(outputs_raw_float);
	print("Converted outputs successfully.");
	print("Loaded outputs successfully.");
	print("Finished loading dataset.");

	Model model;

	DenseLayer layer_hidden(50, true);
	model.addLayer(layer_hidden);

	DenseLayer layer_output(10, true);
	model.addLayer(layer_output);
	model.generateWeights(inputs.get_cols());

	unsigned epochs = 100000;
	float learning_rate = 0.1f;
	unsigned batch_size = 100;
	unsigned batches = inputs.get_rows() / batch_size;

	clock_t batch_time;
	print("Starting training...");
	for (unsigned epoch = 0; epoch < epochs; epoch++)
	{
		float epoch_accuracy = 0;
		time = clock();
		for (unsigned batch = 0; batch < batches; batch++)
		{
			batch_time = clock();
			matrix<float> batch_input = inputs.slice_v(batch * batch_size, (batch + 1) * batch_size);
			matrix<float> batch_output = outputs.slice_v(batch * batch_size, (batch + 1) * batch_size);

			matrix<float> predicted_output = model.feed_forward(batch_input);

			//print_matrix(matrix<float>(28, 28, batch_input.slice_v(0, 1).flatten()));
			print_matrix(predicted_output.slice_v(0, 3));
			print_matrix(batch_output.slice_v(0, 3));

			matrix<float> error = batch_output - predicted_output;
			model.back_propagate(error, batch_input, learning_rate);
			batch_time = clock() - batch_time;

			//float accuracy = calc_error(predicted_output.slicev(0, 1), batch_output.slicev(0, 1)) / batch_size;
			float accuracy = get_num_correct(predicted_output, batch_output) / batch_size;
			epoch_accuracy += accuracy;
			std::cout << "   Batch: " << batch << " took " << ((float)batch_time / CLOCKS_PER_SEC) << " seconds, Accuracy: " << (100.0f * (float)accuracy) << "%" << std::endl;
			/*/if (epoch > 1)
			{
				std::cout << std::endl << std::endl;
				//model.debug();
				//square_error(predicted_output, batch_output, batch_size);
				if (batch > 3)
					exit(1);
			}//*/
		}
		time = clock() - time;

		std::cout << "Epoch: " << epoch << " took " << ((float)time / CLOCKS_PER_SEC) << " seconds, Accuracy: " << 100.0f * (epoch_accuracy / (float) batches) << "%" << std::endl;
	}
}