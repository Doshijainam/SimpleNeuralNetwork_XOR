#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

//init_weights function generates a random weight value 
double init_weights()
{
    return ((double)rand()/(double)RAND_MAX);
}

//sigmoid function takes the argument value as its parameter and returns the calculation of inputvalue x weights + bias
//This is a standard function with proved outputs - Used in forward propogation in the neural network 
double sigmoid(double x)
{
    return 1/(1 + exp(-x)); 
}

//This is derivative of Sigmoid function, Used in backward propogation in the neural network  
double dSigmoid(double x)
{
    return x * (1-x);
}

void shuffle(int *array, size_t n)
{
    if(n>1)
    {
        size_t i;
        for(i = 0;i<n-1;i++)
        {
            size_t j = rand() / (RAND_MAX/(n-i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }

}

int main(void)
{
    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 0.0f},
                                                          {1.0f, 1.0f}};
    
    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                          {1.0f},
                                                          {1.0f},
                                                          {0.0}};

    




}