#include <iostream>
#include <vector>
#include <string>
#include <fstream>



template<typename T>
std::vector<T> LoadAsVector(const std::string& path)
{
	std::fstream _in;
	_in.open(path, std::ios::in | std::ios::binary);

	std::vector<T> temp;
	if (_in.is_open())
	{
		std::streampos _startPos = _in.tellg();
		_in.seekg(0, std::ios::end);
		int _fileSize = _in.tellg() - _startPos;

		const int _tickSize = sizeof(T);
		int _elementCount = _fileSize / _tickSize;

		temp.resize(_elementCount);
		_in.seekg(0);
		_in.read((char*)temp.data(), _fileSize);
		_in.close();
	}
	return temp;
}

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float SigmoidDerivative(float x) { return x * (1.0f - x); }

struct Neuron {
	float value = 0.0f;
	float activation = 0.0f;
	float bias = 0.0f;
	float dC_dA = 0.0f;
	float eBias = 0.0f;
	std::vector<float> weights;
	std::vector<float> eWeight;
};

struct NeuralNetwork {
	NeuralNetwork(std::vector<size_t> layers) {
		for (int i = 0; i < layers.size(); ++i)
			net.push_back(std::vector<Neuron>(layers[i]));

		// target size must match outputLayer size
		target.resize(layers[layers.size() - 1]);

		// neuron weights size have to match number of neurons in previous layer
		// iteration starts form 1 because input layer does not have any previous layer
		for (int lay = 1; lay < net.size(); ++lay)
			for (auto& neuron : net[lay])
			{
				neuron.weights.resize(layers[lay - 1], 1.0f);
				neuron.eWeight.resize(layers[lay - 1], 0.0f);
			}
	}
	void FeedData(std::vector<float>& data) {
		for (int neu = 0; neu < net[0].size() && neu < data.size(); ++neu)
			net[0][neu].activation = Sigmoid(data[neu]);
	}
	void SetTarget(std::vector<float>& targetData)
	{
		auto& outputLayer = net[net.size() - 1];
		target = std::vector<float>(outputLayer.size(), 0.9f);
		for (int neu = 0; neu < outputLayer.size() && neu < targetData.size(); ++neu)
			target[neu] = targetData[neu];
	}
	float Train() {
		// feed forward
		for (size_t currentLayer = 1; currentLayer < net.size(); ++currentLayer)
		{
			for (auto& neuron : net[currentLayer])
			{
				auto& previousLayer = net[currentLayer - 1];
				neuron.value = 0.0f;
				for (int i = 0; i < previousLayer.size(); ++i)
					neuron.value += neuron.weights[i] * previousLayer[i].activation;

				neuron.value /= previousLayer.size();
				neuron.value += neuron.bias;
				neuron.activation = Sigmoid(neuron.value);
				neuron.dC_dA = 0.0f;
			}
		}

		float totalCost = 0.0f;
		auto& outputLayer = net[net.size() - 1];
		for (int i = 0; i < outputLayer.size(); ++i)
		{
			outputLayer[i].dC_dA = 2 * (outputLayer[i].activation - target[i]);
			totalCost += std::pow(outputLayer[i].activation - target[i], 2);
		}

		// propagate backward
		for (int lay = net.size() - 1; lay > 0; --lay)
		{
			auto& currentLayer = net[lay];
			auto& previousLayer = net[lay - 1];
			for (int neu = 0; neu < currentLayer.size(); ++neu)
			{
				auto& currentNeuron = currentLayer[neu];
				float dA_dZ = SigmoidDerivative(currentNeuron.activation);

				float dC_dB = currentNeuron.dC_dA * dA_dZ;
				currentNeuron.eBias += dC_dB;

				for (int wei = 0; wei < currentNeuron.weights.size(); ++wei)
				{
					auto& previousNeuron = previousLayer[wei];
					auto& currentWeight = currentNeuron.weights[wei];
					auto& errorWeight = currentNeuron.eWeight[wei];

					float dZ_dW = previousNeuron.activation;
					float dZ_dPrevA = currentWeight;

					float dC_dW = currentNeuron.dC_dA * dA_dZ * dZ_dW;
					float dC_dPrevA = currentNeuron.dC_dA * dA_dZ * dZ_dPrevA;

					errorWeight += dC_dW;
					previousNeuron.dC_dA += dC_dPrevA;
				}
			}
		}
		return totalCost;
	}
	void Learn(size_t batchSize, float learningRate)
	{
		for (int lay = 1; lay < net.size(); ++lay)
		{
			for (auto& neuron : net[lay])
			{
				neuron.bias -= neuron.eBias / (float)batchSize * learningRate;
				for (int i = 0; i < neuron.weights.size(); ++i)
					neuron.weights[i] -= neuron.eWeight[i] / (float)batchSize * learningRate;
			}
		}
	}

	int Test(std::vector<std::vector<float>>& images, std::vector<uint8_t>& labels)
	{
		auto tempNet = net;
		int successful = 0;
		for (int im = 0; im < images.size(); ++im)
		{
			for (int neu = 0; neu < tempNet[0].size() && neu < images[im].size(); ++neu)
				tempNet[0][neu].activation = Sigmoid(images[im][neu]);

			for (size_t currentLayer = 1; currentLayer < tempNet.size(); ++currentLayer)
			{
				for (auto& neuron : tempNet[currentLayer])
				{
					auto& previousLayer = tempNet[currentLayer - 1];
					neuron.value = 0.0f;
					for (int i = 0; i < previousLayer.size(); ++i)
						neuron.value += neuron.weights[i] * previousLayer[i].activation;

					neuron.value /= previousLayer.size();
					neuron.value += neuron.bias;
					neuron.activation = Sigmoid(neuron.value);
				}
			}

			float a = -99999.0f;
			int decision = -1;
			auto& outputLayer = tempNet[tempNet.size() - 1];
			for (int o = 0; o < outputLayer.size(); ++o)
			{
				if (outputLayer[o].activation > a) {
					a = outputLayer[o].activation;
					decision = o;
				}
			}

			if (decision == labels[im]) ++successful;
		}
		return successful;
	}

	auto Get() { return net; }
	void Wipe() {
		for (auto& lay : net)
			for (auto& neu : lay)
			{
				for (auto& eWei : neu.eWeight)
					eWei = 0.0f;
				neu.eBias = 0.0f;
			}
	}
private:
	std::vector<std::vector<Neuron>> net;
	std::vector<float> target;
};

int main(int argc, char* argv[])
{
	// H means it is entire file with header
	auto train_labels_H = LoadAsVector<uint8_t>("MNIST/train-labels.idx1-ubyte");
	auto train_images_H = LoadAsVector<uint8_t>("MNIST/train-images.idx3-ubyte");
	auto test_labels_H = LoadAsVector<uint8_t>("MNIST/t10k-labels.idx1-ubyte");
	auto test_images_H = LoadAsVector<uint8_t>("MNIST/t10k-images.idx3-ubyte");

	// not rly elegant way to get rid of header :D
	std::vector<uint8_t> train_labels(train_labels_H.begin() + 8, train_labels_H.end());
	std::vector<uint8_t> test_labels(test_labels_H.begin() + 8, test_labels_H.end());

	// organizing pixel data into array of images
	std::vector<std::vector<float>> train_images;
	for (int i = 16; i < train_images_H.size(); ++i)
	{
		static std::vector<float> image;
		image.push_back(train_images_H[i]);
		if ((i - 15) % 784 == 0)
		{
			train_images.push_back(image);
			image.resize(0);
		}
	}
	std::vector<std::vector<float>> test_images;
	for (int i = 16; i < test_images_H.size(); ++i)
	{
		static std::vector<float> image;
		image.push_back(test_images_H[i]);
		if ((i - 15) % 784 == 0)
		{
			test_images.push_back(image);
			image.resize(0);
		}
	}

	// initializing network with 2 hidden layers 16 neuron each and output layer 10 (because of 10 digits)
	NeuralNetwork nn({ 784, 16, 16, 10 });

	// there are 60k images in training MNIST set
	std::vector<float> costs(60000, 0.0f);
	for (;;)
	{
		for (int i = 0; i < 60000; ++i)
		{
			nn.FeedData(train_images[i]);
			std::vector<float> target(10, 0.0f);
			target[train_labels[i]] = 1.0f;
			nn.SetTarget(target);
			costs[i] = nn.Train();
		}
		nn.Learn(60000, 5.0f);
		nn.Wipe();

		float cost_avg = 0.0f;
		for (auto& c : costs) cost_avg += c;
		cost_avg /= 60000.0f;
		std::cout << std::to_string(nn.Test(test_images, test_labels)) + " correctly recognized images ("
			+ std::to_string(cost_avg) + ")\n";
	}
}