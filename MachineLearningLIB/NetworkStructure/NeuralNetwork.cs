using MachineLearningLIB.InitializationFunctions;
using System;
using System.Collections.Generic;

namespace MachineLearningLIB.NetworkStructure
{
    public class NeuralNetwork
    {
        public int ExpectedInputCount { get; }
        public List<NeuralLayer> NeuralLayers { get; }
        public InitializationFunction InitializationFunc { get; }

        private NeuralNetwork(int inputCount, InitializationFunction initializationFunction)
        {
            ExpectedInputCount = inputCount;
            NeuralLayers = new List<NeuralLayer>();
            InitializationFunc = initializationFunction;
        }

        public void Initialize(Random rand)
        {
            for (int i = 0; i < NeuralLayers.Count; i++)
            {
                NeuralLayers[i].ResetWeights(rand);
            }
        }
    }
}
