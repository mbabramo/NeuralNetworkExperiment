using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Results;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.SupervisedLearning.Progress;

namespace NeuralNetworkExperiment
{
    class Program
    {

        public static async Task Main(string[] args)
        {
            int numSamples = 40_000;
            int numSamplesForTraining = 36_000;
            int numSamplesForValidation = 2_000;
            int batchSize = 200;
            int epochs = 1_000;
            Random r = new Random();
            var data = new (float[] X, float[] Y)[numSamples];
            for (int q = 0; q < data.Length; q++)
            {
                float a = (float)r.NextDouble();
                float b = (float)r.NextDouble();
                float a_norm = a; //  2.0f * (a - 0.5f);
                float b_norm = b; // 2.0f * (b - 0.5f);
                data[q].X = new float[] { a_norm, b_norm };
                data[q].Y = new float[] { (float)((a - 0.3)*(a-0.3) + (b-0.7)*(b-0.7)) };
            }
            await TestNeuralNetwork(numSamples, numSamplesForTraining, numSamplesForValidation, batchSize, data, CostFunctionType.Quadratic, 0, TrainingAlgorithms.RMSProp(), epochs);
        }

        private static async Task TestNeuralNetwork(int numSamples, int numSamplesForTraining, int numSamplesForValidation, int batchSize, (float[] X, float[] Y)[] data, CostFunctionType costFunctionType, float dropout, ITrainingAlgorithmInfo trainingAlgorithm, int epochs)
        {
            Stopwatch s = new Stopwatch();
            s.Start();
            INeuralNetwork network = NetworkManager.NewSequential(TensorInfo.Linear(data.First().X.Length),
                NetworkLayers.FullyConnected(50, ActivationType.ReLU),
                NetworkLayers.FullyConnected(1, ActivationType.Tanh, costFunctionType));
            ITrainingDataset trainingData = DatasetLoader.Training(data.Take(numSamplesForTraining), batchSize);
            var validationData = DatasetLoader.Validation(data.Skip(numSamplesForTraining).Take(numSamplesForValidation), 0.005f, 10);
            ITestDataset testData = DatasetLoader.Test(data.Skip(numSamplesForTraining + numSamplesForValidation));
            void TrackBatchProgress(BatchProgress progress)
            {
            }
            TrainingSessionResult trainingResult = await NetworkManager.TrainNetworkAsync(network,
                trainingData,
                trainingAlgorithm,
                epochs, dropout,
                TrackBatchProgress,
                testDataset: testData);
            var lastTrainingReport = trainingResult.TestReports.Last();
            var testDataResults = data.Skip(numSamplesForTraining + numSamplesForValidation).Select(d => (network.Forward(d.X).First(), d.Y.First())).ToList();
            var examples = data.Skip(numSamplesForTraining + numSamplesForValidation).Take(15).Select(d => $"{string.Join(",", d.X)} => {network.Forward(d.X).Single()} (correct: {d.Y.Single()})");
            foreach (var example in examples)
                Console.WriteLine(example);
            var correlation = CorrelationCoefficient(testDataResults.Select(d => d.Item1).ToArray(), testDataResults.Select(d => d.Item2).ToArray());
            var minProjected = testDataResults.Select(d => d.Item1).Min();
            var maxProjected = testDataResults.Select(d => d.Item1).Max();
            var avgAbsolute = testDataResults.Average(d => Math.Abs(d.Item2 - d.Item1));
            Console.WriteLine($"Cost function {costFunctionType} dropout {dropout} trainingAlgorithm {trainingAlgorithm} correlation {correlation} avgabsdiff {avgAbsolute} min-projected {minProjected} max-projected {maxProjected} time {s.ElapsedMilliseconds / 1000.0 }");
        }
        static float CorrelationCoefficient(float[] values1, float[] values2)
        {
            if (values1.Length != values2.Length)
                throw new ArgumentException("values must be the same length");

            var avg1 = values1.Average();
            var avg2 = values2.Average();

            var sum1 = values1.Zip(values2, (x1, y1) => (x1 - avg1) * (y1 - avg2)).Sum();

            var sumSqr1 = values1.Sum(x => Math.Pow((x - avg1), 2.0));
            var sumSqr2 = values2.Sum(y => Math.Pow((y - avg2), 2.0));

            var result = sum1 / Math.Sqrt(sumSqr1 * sumSqr2);

            return (float)result;
        }
    }
}
