# ArtificialNeuralNetwork
C++ class for artificial feed-forward neural networks.

### Disclaimer
This is not production-worthy code! View this simply as a proof-of-concept.

### Initialization
```C++
NeuralNetwork(vector<int> sizes, void(*Evaluate)(NeuralNetwork *, NeuralData));
NeuralNetwork(string filename, void(*Evaluate)(NeuralNetwork *, NeuralData));
```

### Feed Forward
```C++
Col<double> FeedForward(Col<double> a);
```

### Stochastic Gradient Descent
```C++
void StochasticGradientDescent(NeuralData trainingData, int epochs, int miniBatchSize, double eta, NeuralData testData);
```

### Save
```C++
void Save(string filename);
```

### Open
```C++
void Open(string filename);
```

### Setters/Getters
```C++
int GetEpoch();
void SetSizes(vector<int> sizes);
```

### Typedefs
```C++
typedef vector<vector<Col<double>>> NeuralData;
```

### Example
```C++
#include "ArtificialNeuralNetwork.h"
#include <vector>
#include <fstream>

#define INPUT_NEURONS               1
#define HIDDEN_NEURONS            100
#define OUTPUT_NEURONS              1

#define EPOCHS                   1000
#define MINI_BATCH_SIZE            10
#define LEARNING_RATE             1.2

#define TRAINING_DATA_SIZE       1000
#define TEST_DATA_SIZE            100

#define TRAINING_DATA_RANGE  0.2, 0.8
#define TEST_DATA_RANGE      0.0, 1.0

#define TRAINING_OUTPUT "training.dat"
#define TEST_OUTPUT         "test.dat"

#define PI 3.1415926

using namespace std;
using namespace arma;

int trial = 0;

//Function provided to show how to use Evaluate function
void Evaluate(NeuralNetwork * nn, NeuralData testData)
{
	ofstream file("results.dat");

	for (unsigned int testExample = 0; testExample < testData.size(); testExample++)
	{
		Col<double> result = nn->FeedForward(testData[testExample][0]);

		file << testData[testExample][0][0] << " " << result[0] << endl;
	}

	file.close();

	stringstream ss;

	ss << "gnuplot -e \"set terminal png size 750, 500; set output 'trial" << ++trial << ".png'; set xr[0:1]; set yr[0:1]; set nokey; plot 'training.dat', 0.5 + 0.25 * cos(4.0 * pi * x), 'results.dat';\"";

	system(ss.str().c_str());

	return;
}

default_random_engine randomEngine;

double RandomDouble(double lower, double upper)
{
	uniform_real_distribution<double> unif(lower, upper);
	return unif(randomEngine);
}

//Function provided to show format of test/training data
NeuralData GenerateData(int samples, string filename, double from, double to)
{
	ofstream file(filename);

	NeuralData output;
	for (int sample = 0; sample < samples; sample++)
	{
		output.push_back(vector<Col<double>>());

		vector<double> x;
		x.push_back(RandomDouble(from, to));
		file << x.back() << " ";

		vector<double> y;
		y.push_back(0.5 + 0.25 * cos(4.0 * PI * x.back()));
		file << y.back() << endl;

		output.back().push_back(Col<double>(x));
		output.back().push_back(Col<double>(y));
	}

	file.close();

	return output;
}

int main()
{
	vector<int> sizes = { INPUT_NEURONS, HIDDEN_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS };

	NeuralNetwork nn(sizes, &Evaluate);

	NeuralData trainingData = GenerateData(TRAINING_DATA_SIZE, TRAINING_OUTPUT, TRAINING_DATA_RANGE);
	NeuralData testData = GenerateData(TEST_DATA_SIZE, TEST_OUTPUT, TEST_DATA_RANGE);

	nn.StochasticGradientDescent(trainingData, EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, testData);

	return 0;
}
```
