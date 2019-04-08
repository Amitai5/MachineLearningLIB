using MachineLearningLIB.ActivationFunctions;
using MachineLearningLIB.InitializationFunctions;
using MachineLearningLIB.NetworkStructure;
using System;

namespace MachineLearningLIB.NetworkBuilder
{
    public interface INeuralNetworkBuilderSetInitMethod
    {
        INeuralNetworkBuilderCreateInput SetInitMethod(InitializationFunction initializationFunction);
    }

    public interface INeuralNetworkBuilderCreateInput
    {
        INeuralNetworkBuilderCreateLayers CreateInputLayer(int inputCount);
    }

    public interface INeuralNetworkBuilderCreateLayers
    {
        INeuralNetworkBuilderFinal CreateOutputLayer(int neuronCount, ActivationFunc activationFunc);
        INeuralNetworkBuilderCreateLayers AddHiddenLayer(int neuronCount, ActivationFunc activationFunc);
    }

    public interface INeuralNetworkBuilderFinal
    {
        NeuralNetwork Build(Random rand);
        //GeneticNeuralNetwork[] BuildMany(Random rand, int networkCount);
    }
}
