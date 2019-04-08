using MachineLearningLIB.ActivationFunctions;
using MachineLearningLIB.InitializationFunctions;
using MachineLearningLIB.NetworkStructure;
using System;

namespace LibraryTester
{
    static class Program
    {
        private static void Main(string[] args)
        {
            //Set Test Data
            double[,] TestDataInputs = new double[,]
            {
                //Each column pertains to another set of input data
                { 1, 1 },
                { 1, 0 },
                { 0, 0 }
            };

            NeuralLayer layer = new NeuralLayer(new Sigmoid(), InitializationFunction.Random, 2, 1);
            layer.ResetWeights(new Random());
            layer.ResetBiases(new Random());

            Tensor inputTensor = new Tensor(TestDataInputs);
            Tensor temp = layer.Compute(inputTensor);
        }
    }
}
