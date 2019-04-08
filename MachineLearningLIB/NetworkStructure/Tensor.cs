using MathNet.Numerics.LinearAlgebra;
using System;
using System.Diagnostics;

namespace MachineLearningLIB.NetworkStructure
{
    [DebuggerDisplay("Tensor: {RowCount}x{ColCount}")]
    public class Tensor
    {
        public double this[int x, int y]
        {
            get
            {
                return tensorBackend[x, y];
            }

            set
            {
                tensorBackend[x, y] = value;
            }
        }
        private readonly Matrix<double> tensorBackend;

        public long RowCount => tensorBackend.RowCount;
        public long ColCount => tensorBackend.ColumnCount;
        public double[,] Values => tensorBackend.ToArray();

        public Tensor(double[,] tensorValues, bool separateTestsByRow = true)
        {
            tensorBackend = CreateMatrix.SparseOfArray(tensorValues);

            //Check if each row is a separate set of input data
            if (separateTestsByRow)
            {
                tensorBackend = tensorBackend.Transpose();
            }
        }

        private Tensor(Matrix<double> tensorValues)
        {
            tensorBackend = tensorValues;
        }

        #region Tensor Operators
        public static Tensor operator -(Tensor t1, Tensor t2)
        {
            //Check if the matrices are the same size
            if (t1.RowCount != t2.RowCount || t1.ColCount != t2.ColCount)
            {
                throw new InvalidOperationException($"The matrices are not the same size! {Environment.NewLine}" +
                    $"Left Matrix: ({t1.RowCount}, {t1.ColCount}) {Environment.NewLine}Right Matrix: ({t2.RowCount}, {t2.ColCount})");
            }

            //Return the difference between the two matrices
            return new Tensor(t1.tensorBackend.Subtract(t2.tensorBackend));
        }

        public static Tensor operator +(Tensor t1, Tensor t2)
        {
            //Check if the matrices are the same size
            if (t1.RowCount != t2.RowCount && t1.ColCount != t2.ColCount)
            {
                throw new InvalidOperationException($"The matrices are not the same size! {Environment.NewLine}" +
                    $"Left Matrix: ({t1.RowCount}, {t1.ColCount}) {Environment.NewLine}Right Matrix: ({t2.RowCount}, {t2.ColCount})");
            }

            //Create new tensor back-end
            double[,] newTensorBackend = new double[t1.RowCount, t1.ColCount];

            //Get the row sums
            Vector<double> t2RowSums = t2.tensorBackend.RowSums();

            //Go through t1 and add t2
            for (int i = 0; i < t1.RowCount; i++)
            {
                for (int j = 0; j < t1.ColCount; j++)
                {
                    newTensorBackend[i, j] = t1[i, j] + t2RowSums[i];
                }
            }

            //Return the new sum
            return new Tensor(newTensorBackend);
        }

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            //Check if matrix multiplication is possible
            if (t1.ColCount != t2.RowCount)
            {
                throw new InvalidOperationException($"T1 = ({t1.RowCount}, {t1.ColCount}) T2 = ({t2.RowCount}, {t2.ColCount})");
            }

            //Return the product of the two matrices
            return new Tensor(t1.tensorBackend.Multiply(t2.tensorBackend));
        }
        #endregion Tensor Operators
    }
}
