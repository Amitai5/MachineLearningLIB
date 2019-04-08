using MachineLearningLIB.ActivationFunctions;
using MachineLearningLIB.InitializationFunctions;
using System;

namespace MachineLearningLIB.NetworkStructure
{
    public class NeuralLayer
    {
        private readonly Tensor biasWeights;
        private readonly Tensor neuronWeights;

        public long NeuronCount { get; }
        public ActivationFunc ActivationFunc { get; }
        public InitializationFunction InitializationFunc { get; }

        public NeuralLayer(ActivationFunc activationFunc, InitializationFunction initFunc, long inputCount, long neuronCount)
        {
            //Set layer globals
            NeuronCount = neuronCount;
            InitializationFunc = initFunc;
            ActivationFunc = activationFunc;

            //Initialize neuron weight matrix values
            double[,] weightValues = new double[neuronCount, inputCount];

            //Initialize neuron bias matrix values
            double[,] biasValues = new double[neuronCount, weightValues.GetLength(1)];

            //Create tensors
            biasWeights = new Tensor(biasValues, false);
            neuronWeights = new Tensor(weightValues, false);
        }

        public Tensor Compute(Tensor inputs)
        {
            Tensor temp = (neuronWeights * inputs) + biasWeights;
            double[,] newTensor = new double[temp.RowCount, temp.ColCount];

            //Run the activation function
            for (int i = 0; i < temp.RowCount; i++)
            {
                for (int j = 0; j < temp.ColCount; j++)
                {
                    newTensor[i, j] = ActivationFunc.Function(temp.Values[i, j]);
                }
            }
            return new Tensor(newTensor);
        }

        public void ResetWeights(Random rand)
        {
            for (int i = 0; i < neuronWeights.RowCount; i++)
            {
                for (int j = 0; j < neuronWeights.ColCount; j++)
                {
                    neuronWeights[i, j] = DendriteInitialization.GetInitFunction(InitializationFunc).Invoke(ActivationFunc, rand);
                }
            }
        }

        public void ResetBiases(Random rand)
        {
            for (int i = 0; i < biasWeights.RowCount; i++)
            {
                for (int j = 0; j < biasWeights.ColCount; j++)
                {
                    biasWeights[i, j] = DendriteInitialization.GetInitFunction(InitializationFunc).Invoke(ActivationFunc, rand);
                }
            }
        }
    }
}
