# ArtificialNeuralNetwork
C++ class for artificial feed-forward neural networks.

### Disclaimer
This is not production-worthy code! View this simply as a proof-of-concept.

### Initialization
```C++
NeuralNetwork(vector<int> sizes, void(*Evaluate)(NeuralNetwork *, NeuralData));
NeuralNetwork(string filename, void(*Evaluate)(NeuralNetwork *, NeuralData));
```
A `NeuralNetwork` can be initialized two different ways:

 1. With a vector containing the number of neurons in each layer and a pointer to an `Evaluate` function
 2. With a filename of a saved `NeuralNetwork` and a pointer to an `Evaluate` function.
 3. With a vector containing the number of neurons in each layer.
 4. With a filename of a saved `NeuralNetwork`.
 
The constructors without an `Evaluate` function pointer will be unable to evaluate test data in the `StochasticGradientDescent` method. Additionally, the saved `NeuralNetwork` file must have been generated by the `Save` method.

### Feed Forward
```C++
NeuralOutput FeedForward(NeuralInput a);
```
This method takes an input to the `NeuralNetwork` and passes it all the way through resulting in an outpu from the entire network. The input must match the dimension of the first layers specified by `sizes` in the constructors or the `SetSizes` method and the output will match the dimension of the output layer specified in the same way.

### Stochastic Gradient Descent
```C++
void StochasticGradientDescent(NeuralData trainingData, int epochs, int miniBatchSize, double eta, NeuralData testData);
```
This method trains the `NeuralNetwork` and optionally evaluates the performance of the network after each epoch using the user-defined `Evaluate` function. For more information about the stochastic gradient descent algorithm, visit the [Wikipedia page](https://en.wikipedia.org/wiki/Stochastic_gradient_descent);

### Save
```C++
void Save(string filename);
```
This method saves the `NeuralNetwork`'s most recent weights and biases and layer dimensions in a binary file. This should be universal, not compiler-specific, as the information is iterated through.

### Open
```C++
void Open(string filename);
```
This method opens a `NeuralNetwork` binary file containing the weights, biases, and layer dimensions generated by the `Save` method. This should be universal, not compiler specific, as the information is iterated through.

### Setters/Getters
```C++
int GetEpoch();
void SetSizes(vector<int> sizes);
```
To allow an `Evaluate` function to know what the most recent epoch completed is, the `GetEpoch` method is provided. Also, to allow altering of the layer dimensions without declaring a new `NeuralNetwork`, the `SetSizes` method is provided. All information about the `NeuralNetwork` will be cleared.

### Typedefs
```C++
typedef vector<vector<Col<double>>> NeuralData;
typedef vector<Col<double>> NeuralBiases;
typedef vector<Mat<double>> NeuralWeights;
typedef Col<double> NeuralOutput;
typedef Col<double> NeuralInput;
typedef Col<double> NeuralError;
typedef vector<Col<double>> NeuralDataPoint;
```
Since the format of the information passed into the `NeuralNetwork` has a lengthy type and can mean completely different things depending on how they are used, these types are defined. 

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
		NeuralOutput result = nn->FeedForward(testData[testExample][0]);

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
		output.push_back(NeuralDataPoint());

		vector<double> x;
		x.push_back(RandomDouble(from, to));
		file << x.back() << " ";

		vector<double> y;
		y.push_back(0.5 + 0.25 * cos(4.0 * PI * x.back()));
		file << y.back() << endl;

		output.back().push_back(NeuralInput(x));
		output.back().push_back(NeuralOutput(y));
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
This is a simple example that shows the algortihm in action by learning a sine wave. The information is displayed visually by plotting the data using GNUPlot after each epoch. (You must have GNUPlot in you `PATH` in order for this program to run properly.) All necessary information should be provided in this example showing how to format training/test data and how to structure an `Evaluate` function.
