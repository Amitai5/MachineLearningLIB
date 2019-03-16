using MachineLearningLIB.ActivationFunctions;
using MachineLearningLIB.InitializationFunctions;
using MachineLearningLIB.LearningAlgorithms;
using MachineLearningLIB.NetworkBuilder;
using MachineLearningLIB.NetworkStructure;
using System;

namespace LibraryTester
{
    static class Program
    {
        private static void Main(string[] args)
        {
            NeuralNetwork Network = NeuralNetworkBuilder.StartBuild()
                .SetInitMethod(InitializationFunction.Random)
                .CreateInputLayer(2)
                .AddHiddenLayer(2, new Sigmoid())
                .CreateOutputLayer(1, new Sigmoid())
                .Build(new Random());

            //Set Test Data
            double[][] TestDataOutputs = new double[][]
            {
                new double[]{ 0 },
                new double[]{ 1 },
                new double[]{ 1 },
                new double[]{ 0 }
            };
            double[][] TestDataInputs = new double[][]
            {
                new double[]{ 0, 0 },
                new double[]{ 1, 0 },
                new double[]{ 0, 1 },
                new double[]{ 1, 1 }
            };

            double Error = 0;
            Backpropagation Backprop = new Backpropagation(Network);
            while (Backprop.EpochCount < 8000)
            {
                Error = Backprop.TrainEpoch(TestDataInputs, TestDataOutputs);
            }
        }
    }
}
