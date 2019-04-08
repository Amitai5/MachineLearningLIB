using MachineLearningLIB.ActivationFunctions;
using MachineLearningLIB.InitializationFunctions;
using MachineLearningLIB.NetworkStructure;
using System;

namespace MachineLearningLIB.NetworkBuilder
{
    public sealed class NeuralNetworkBuilder : INeuralNetworkBuilderSetInitMethod, INeuralNetworkBuilderCreateInput, INeuralNetworkBuilderCreateLayers, INeuralNetworkBuilderFinal
    {
        private int InputCount = 0;
        private NeuralNetwork BuildingNet;
        private InitializationFunction InitializationFunction;

        private NeuralNetworkBuilder()
        {

        }

        public static INeuralNetworkBuilderSetInitMethod StartBuild()
        {
            return new NeuralNetworkBuilder();
        }

        public INeuralNetworkBuilderCreateLayers CreateInputLayer(int inputCount)
        {
            //BuildingNet = new NeuralNetwork(inputCount, InitializationFunction);
            InputCount = inputCount;
            return this;
        }

        public INeuralNetworkBuilderCreateInput SetInitMethod(InitializationFunction initializationFunction)
        {
            InitializationFunction = initializationFunction;
            return this;

        }

        #region Add Layers

        public INeuralNetworkBuilderCreateLayers AddHiddenLayer(int neuronCount, ActivationFunc activationFunc)
        {
            long previousLayerNeuronCount = BuildingNet.NeuralLayers.Count == 0 ? InputCount : BuildingNet.NeuralLayers[BuildingNet.NeuralLayers.Count - 1].NeuronCount;
            BuildingNet.NeuralLayers.Add(new NeuralLayer(activationFunc, InitializationFunction, previousLayerNeuronCount, neuronCount));
            return this;
        }

        public INeuralNetworkBuilderFinal CreateOutputLayer(int neuronCount, ActivationFunc activationFunc)
        {
            long previousLayerNeuronCount = BuildingNet.NeuralLayers.Count == 0 ? InputCount : BuildingNet.NeuralLayers[BuildingNet.NeuralLayers.Count - 1].NeuronCount;
            BuildingNet.NeuralLayers.Add(new NeuralLayer(activationFunc, InitializationFunction, previousLayerNeuronCount, neuronCount));
            return this;
        }

        #endregion Add Layers


        #region Final Build

        public NeuralNetwork Build(Random rand)
        {
            BuildingNet.Initialize(rand);
            return BuildingNet;
        }

        #endregion Final Build
    }
}
