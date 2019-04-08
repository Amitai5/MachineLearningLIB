using MachineLearningLIB.ActivationFunctions;
using System;

namespace MachineLearningLIB.InitializationFunctions
{
    public static class DendriteInitialization
    {
        public static Func<ActivationFunc, Random, double> GetInitFunction(InitializationFunction initializationFunction)
        {
            switch (initializationFunction)
            {
                case InitializationFunction.One:
                    return OneInitialization;
                default:
                    return RandomInitialization;
            }
        }

        public static double RandomInitialization(ActivationFunc activationFunc, Random rand)
        {
            return rand.NextDouble(activationFunc.DendriteMinGen, activationFunc.DendriteMaxGen);
        }

        public static double OneInitialization(ActivationFunc activationFunc, Random rand)
        {
            return 1;
        }
    }
}
