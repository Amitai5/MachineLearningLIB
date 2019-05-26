using MachineLearningLIB.ActivationFunctions;
using MachineLearningLIB.InitializationFunctions;
using MachineLearningLIB.NetworkBuilder;
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
                { 0, 1 },
                { 0, 0 }
            };
            Tensor inputTensor = new Tensor(TestDataInputs);

            //Create network
            NeuralNetwork network = NeuralNetworkBuilder.StartBuild()
                .SetInitMethod(InitializationFunction.Random)
                .CreateInputLayer(2)
                .AddHiddenLayer(2, new Sigmoid())
                .CreateOutputLayer(1, new Sigmoid())
                .Build(new Random());

            //Compute network values
            Tensor output = network.Compute(inputTensor);
        }
    }
}
