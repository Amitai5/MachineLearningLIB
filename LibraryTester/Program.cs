using NeuralNetLIB.ActivationFunctions;
using NeuralNetLIB.InitializationFunctions;
using NeuralNetLIB.LearningAlgorithms;
using NeuralNetLIB.NetworkStructure;
using NeuralNetLIB.NetworkStructure.NetworkBuilder;
using System;

namespace LibraryTester
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork Network = new NeuralNetworkBuilder(InitializationFunction.Random)
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
