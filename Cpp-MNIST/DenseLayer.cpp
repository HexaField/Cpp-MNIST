#include "pch.h"
#include "DenseLayer.h"


DenseLayer::DenseLayer(int neurons, bool bias)
{
	num_neurons = neurons;
	has_bias = bias;
}

DenseLayer::~DenseLayer()
{
}
