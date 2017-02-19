#ifndef ARTIFICIAL_NEURAL_NETWORK_HEADER
#define ARTIFICIAL_NEURAL_NETWORK_HEADER

#include <armadillo>    //Mat, Col, set_seed()
#include <vector>		//vector, swap()
#include <random>		//default_random_engine, uniform_int_distribution
#include <chrono>		//system_clock
#include <fstream>		//ifstream, ofstream

#pragma comment(lib, "lapack_win64_MT.lib")
#pragma comment(lib, "blas_win64_MT.lib")

using namespace std;
using namespace arma;

Col<double> ActiviationFunction(Col<double> z)
{
	return 1.0 / (1.0 + exp(-z));
}
Col<double> ActivationFunctionPrime(Col<double> z)
{
	return ActiviationFunction(z) % (1 - ActiviationFunction(z));
}

class NeuralNetwork
{
public:
	NeuralNetwork(vector<int> sizes, void(*Evaluate)(NeuralNetwork *, vector<vector<Col<double>>>));
	NeuralNetwork(string filename, void(*Evaluate)(NeuralNetwork *, vector<vector<Col<double>>>));
	
	Col<double> FeedForward(Col<double> a);
	void StochasticGradientDescent(vector<vector<Col<double>>> trainingData, int epochs, int miniBatchSize, double eta, vector<vector<Col<double>>> testData);

	void Save(string filename);
	void Open(string filename);
	
	int GetEpoch();
	void SetSizes(vector<int> sizes);

private:
	int Epoch;
	int NumberOfLayers;
	vector<int> sizes;
	vector<Col<double>> b;
	vector<Mat<double>> w;
	vector<Col<double>> empty_b;
	vector<Mat<double>> empty_w;

	void(*Evaluate)(NeuralNetwork *, vector<vector<Col<double>>>);

	void Backpropogation(Col<double> x, Col<double> y, vector<Col<double>> * delta_nabla_b, vector<Mat<double>> * delta_nabla_w);

	vector<vector<vector<Col<double>>>> SplitIntoMiniBatches(vector<vector<Col<double>>> trainingData, int miniBatchSize);
};

NeuralNetwork::NeuralNetwork(vector<int> sizes, void(*Evaluate)(NeuralNetwork *, vector<vector<Col<double>>>))
{
	SetSizes(sizes);

	this->Evaluate = Evaluate;
	Epoch = 0;
}
NeuralNetwork::NeuralNetwork(string filename, void(*Evaluate)(NeuralNetwork *, vector<vector<Col<double>>>))
{
	Open(filename);

	this->Evaluate = Evaluate;
	Epoch = 0;
}

Col<double> NeuralNetwork::FeedForward(Col<double> a)
{
	for (unsigned int l = 0; l < b.size(); l++)
		a = ActiviationFunction((w[l] * a) + b[l]);

	return a;
}
void NeuralNetwork::StochasticGradientDescent(vector<vector<Col<double>>> trainingData, int epochs, int miniBatchSize, double eta, vector<vector<Col<double>>> testData)
{
	Epoch = 0;

	if (testData.size() > 0)
		Evaluate(this, testData);

	int n = (int)trainingData.size();
	for (Epoch = 1; Epoch <= epochs; Epoch++)
	{
		vector<vector<vector<Col<double>>>> miniBatches = SplitIntoMiniBatches(trainingData, miniBatchSize);

		for (unsigned int miniBatch = 0; miniBatch < miniBatches.size(); miniBatch++)
		{
			vector<Col<double>> nabla_b = empty_b;
			vector<Mat<double>> nabla_w = empty_w;

			for (unsigned int trainingExample = 0; trainingExample < miniBatches[miniBatch].size(); trainingExample++)
			{
				vector<Col<double>> delta_nabla_b = empty_b;
				vector<Mat<double>> delta_nabla_w = empty_w;

				Backpropogation(miniBatches[miniBatch][trainingExample][0], miniBatches[miniBatch][trainingExample][1], &delta_nabla_b, &delta_nabla_w);

				for (unsigned int l = 0; l < nabla_b.size(); l++)
				{
					nabla_b[l] = nabla_b[l] + delta_nabla_b[l];
					nabla_w[l] = nabla_w[l] + delta_nabla_w[l];
				}
			}

			for (unsigned int l = 0; l < b.size(); l++)
			{
				b[l] = b[l] - (eta / miniBatches[miniBatch].size()) * nabla_b[l];
				w[l] = w[l] - (eta / miniBatches[miniBatch].size()) * nabla_w[l];
			}
		}

		if (testData.size() > 0)
			Evaluate(this, testData);
	}
}

void NeuralNetwork::Open(string filename)
{
	ifstream ifile(filename, ios::binary);

	ifile.read((char *)&NumberOfLayers, sizeof(int));

	for (int l = 0; l < NumberOfLayers; l++)
	{
		int temp;
		ifile.read((char *)&temp, sizeof(int));
		sizes.push_back(temp);
	}

	for (int l = 0; l < NumberOfLayers - 1; l++)
	{
		b.push_back(Col<double>(sizes[l + 1]));
		for (int j = 0; j < sizes[l + 1]; j++)
			ifile.read((char *)&b[l](j), sizeof(double));
		empty_b.push_back(Col<double>(sizes[l + 1], fill::zeros));
	}

	for (int l = 0; l < NumberOfLayers - 1; l++)
	{
		w.push_back(Mat<double>(sizes[l + 1], sizes[l]));
		for (int j = 0; j < sizes[l + 1]; j++)
			for (int k = 0; k < sizes[l]; k++)
				ifile.read((char *)&w[l](j, k), sizeof(double));
		empty_w.push_back(Mat<double>(sizes[l + 1], sizes[l], fill::zeros));
	}

	ifile.close();
}
void NeuralNetwork::Save(string filename)
{
	ofstream ofile(filename, ios::binary);

	ofile.write((char *)&NumberOfLayers, sizeof(int));

	for (int l = 0; l < NumberOfLayers; l++)
		ofile.write((char *)&sizes[l], sizeof(int));

	for (int l = 0; l < NumberOfLayers - 1; l++)
		for (int j = 0; j < sizes[l + 1]; j++)
			ofile.write((char *)&b[l](j), sizeof(double));

	for (int l = 0; l < NumberOfLayers - 1; l++)
		for (int j = 0; j < sizes[l + 1]; j++)
			for (int k = 0; k < sizes[l]; k++)
				ofile.write((char *)&w[l](j, k), sizeof(double));

	ofile.close();
}

int NeuralNetwork::GetEpoch()
{
	return Epoch;
}
void NeuralNetwork::SetSizes(vector<int> sizes)
{
	arma_rng::set_seed((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());

	NumberOfLayers = (int)sizes.size();
	this->sizes = sizes;

	b.clear();
	empty_b.clear();
	w.clear();
	empty_w.clear();

	for (int l = 1; l < NumberOfLayers; l++)
	{
		b.push_back(Col<double>(sizes[l], fill::randn));
		empty_b.push_back(Col<double>(sizes[l], fill::zeros));
		w.push_back(Mat<double>(sizes[l], sizes[l - 1], fill::randn));
		empty_w.push_back(Mat<double>(sizes[l], sizes[l - 1], fill::zeros));
	}
}

void NeuralNetwork::Backpropogation(Col<double> x, Col<double> y, vector<Col<double>> * delta_nabla_b, vector<Mat<double>> * delta_nabla_w)
{
	Col<double> a = x;
	vector<Col<double>> as; as.push_back(a);
	Col<double> z;
	vector<Col<double>> zs;

	for (unsigned int l = 0; l < b.size(); l++)
	{
		z = (w[l] * a) + b[l];
		zs.push_back(z);
		a = ActiviationFunction(z);
		as.push_back(a);
	}

	Col<double> delta = (as[as.size() - 1] - y) % ActivationFunctionPrime(zs[zs.size() - 1]);
	(*delta_nabla_b)[delta_nabla_b->size() - 1] = delta;
	(*delta_nabla_w)[delta_nabla_w->size() - 1] = delta * as[as.size() - 2].t();

	for (int l = (int)delta_nabla_b->size() - 2; l >= 0; l--)
	{
		delta = (w[l + 1].t() * delta) % ActivationFunctionPrime(zs[l]);
		(*delta_nabla_b)[l] = delta;
		(*delta_nabla_w)[l] = delta * as[l].t();
	}
}

vector<vector<vector<Col<double>>>> NeuralNetwork::SplitIntoMiniBatches(vector<vector<Col<double>>> trainingData, int miniBatchSize)
{
	std::default_random_engine random;
	random.seed((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<int> discrete(0, (int)trainingData.size() - 1);

	vector<vector<vector<Col<double>>>> Output;

	for (unsigned int index = 0; index < trainingData.size() - 1; index++)
		swap(trainingData[discrete(random)], trainingData[index]);

	int NumberOfMiniBatches = (int)trainingData.size() / miniBatchSize;
	for (int MiniBatchIndex = 0; MiniBatchIndex < NumberOfMiniBatches; MiniBatchIndex++)
	{
		Output.push_back(vector<vector<Col<double>>>());
		for (int i = MiniBatchIndex; i < MiniBatchIndex + miniBatchSize; i++)
			Output[MiniBatchIndex].push_back(trainingData[i]);
	}

	return Output;
}

#endif
