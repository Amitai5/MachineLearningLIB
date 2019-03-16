using MachineLearningLIB.NetworkStructure;
using System;
using System.Threading.Tasks;

namespace MachineLearningLIB.LearningAlgorithms
{
    public class Genetics
    {
        //Store The Neural Net Data
        private readonly GeneticNeuralNetwork[] NeuralNets;
        public NeuralNetwork BestNetwork { get; private set; }
        public double BestNetworkFitness => NeuralNets[0].Fitness;

        //Store The Training Data
        public Random Rand { get; }
        public double MutationRate { get; }
        public long GenerationCount { get; private set; }

        public Genetics(Random rand, GeneticNeuralNetwork[] geneticNetworkPopulation, double mutationRate = 0.05)
        {
            //Store Neural Network Data
            Rand = rand;
            MutationRate = mutationRate;

            //Store The Neural Networks
            NeuralNets = geneticNetworkPopulation;
        }

        public void TrainGeneration(double[][] inputs, double[][] outputs)
        {
            //Cross Over 80% Of Nets & Randomize 10%
            int OneTenthPopulation = NeuralNets.Length / 10;
            for (int i = OneTenthPopulation; i < NeuralNets.Length; i++)
            {
                GeneticNeuralNetwork CurrentNet = NeuralNets[i];
                if (i < 9 * OneTenthPopulation)
                {
                    CurrentNet.CrossOverAndMutate(NeuralNets[i % OneTenthPopulation], MutationRate, Rand);
                }
                else
                {
                    CurrentNet.Initialize(Rand);
                }
            }

            //Calculate Fitnesses & Sort
            Parallel.For(0, NeuralNets.Length, j => CalculateFitness(NeuralNets[j], inputs, outputs));
            Array.Sort(NeuralNets, (a, b) => a.Fitness.CompareTo(b.Fitness));
            BestNetwork = NeuralNets[0];
            GenerationCount++;
        }

        private void CalculateFitness(GeneticNeuralNetwork neuralNetwork, double[][] inputs, double[][] desiredOutputs)
        {
            double MeanAbsoluteError = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                double Output = neuralNetwork.Compute(inputs[i])[0];
                MeanAbsoluteError += Math.Pow(desiredOutputs[i][0] - Output, 2);
            }
            neuralNetwork.Fitness = MeanAbsoluteError / inputs.Length;
        }
    }
}
