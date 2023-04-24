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
    
    //Now initialize the random weights value between Input Layer and the Hidden Layer

    for(int i = 0;i<numInputs;i++)
    {
        for(int j = 0; j<numHiddenNodes;j++)
        {
            hiddenWeights[i][j] = init_weights();
        }
    }

    //Now initialize the random weights value between Hidden Layer and the Output Layer

    for(int i = 0;i<numHiddenNodes;i++)
    {
        for(int j = 0; j<numOutputs;j++)
        {
            outputWeights[i][j] = init_weights();
        }
    }

    //Now initialize the Bias value between the hidden Layer and the output Layer

    for(int i = 0;i<numOutputs;i++)
    {
        outputLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0,1,2,3};

    int numberOfEpochs = 10000;

    //Train the neural network

    for(int epoch = 0; epoch<numberOfEpochs; epoch++)
    {
        shuffle(trainingSetOrder,numTrainingSets);

        for(int x = 0; x<numTrainingSets;x++)
        {
            int i = trainingSetOrder[x];

        //Forward Pass
        //Compute the Hidden Layer Activation 
        //From Inputs to Hidden Layer
        for(int j = 0; j<numHiddenNodes;j++)
        {
            double activation = hiddenLayerBias[j];

            for(int k = 0;k<numInputs;k++)
            {
                activation += training_inputs[i][k] * hiddenWeights[k][j]; 
            }
            //Updating the Hidden layer
            hiddenLayer[j] = sigmoid(activation);
            
        }

        //From Hidden Layer to Outputs
        for(int j = 0; j<numOutputs;j++)
        {
            double activation = outputLayerBias[j];

            for(int k = 0;k<numHiddenNodes;k++)
            {
                activation += hiddenLayer[k] * outputWeights[k][j]; 
            }
            //Updating the Hidden layer
            outputLayer[j] = sigmoid(activation);
            
        }

        printf("Input : %g %g   Output : %g    Predicted Output: %g\n",training_inputs[i][0],training_inputs[i][1],outputLayer[0],training_outputs[i][0]);


        //This computes the change and adjusts the error in the weights of the  output layer while back propogation 
        double deltaOutput[numOutputs];

        for(int j = 0;j<numOutputs;j++)
        {
            double error = ((training_outputs[i][j]) - outputLayer[j]);
            deltaOutput[j] = error * dSigmoid(outputLayer[j]); // Delta output stores the errors for all the combinations
        }

        //This computes the change and adjusts the error in the weights of the hidden layer while back propogation 

        double deltaHidden[numHiddenNodes];

        for(int j = 0; j<numHiddenNodes;j++)
        {
            double error = 0.0f;
            for(int k = 0 ; k < numOutputs;k++)
            {
                error += deltaOutput[k] * outputWeights[j][k]; //This step multiplies the delta output with the actual outputweights    
            }
            deltaHidden[j] = error * dSigmoid(hiddenLayer[j]); //This step adjusts the error 
        }

        //Apply the changes to the output weights 
        for(int j = 0; j<numOutputs;j++)
        {
            outputLayerBias[j] += deltaOutput[j] * lr;
            for(int k = 0; k<numHiddenNodes; k++)
            {
                outputWeights[k][j] = hiddenLayer[k] * deltaOutput[j] * lr; 
            }
        }

        //Apply the changes to the Hidden weights 
        for(int j = 0; j<numHiddenNodes;j++)
        {
            hiddenLayerBias[j] += deltaHidden[j] * lr;
            for(int k = 0; k<numInputs; k++)
            {
                hiddenWeights[k][j] = training_inputs[i][k] * deltaHidden[j] * lr; 
            }
        }





        }


    }

    




}