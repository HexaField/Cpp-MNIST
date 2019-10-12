#pragma once
class DenseLayer
{
public:
	DenseLayer(int neurons, bool bias);
	int get_num_neurons() { return num_neurons; }
	bool get_has_bias() { return has_bias; }
	~DenseLayer();
private:
	int num_neurons;
	bool has_bias;

};

