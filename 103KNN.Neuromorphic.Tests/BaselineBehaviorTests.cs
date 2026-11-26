using System;
using System.IO;
using _103KNN.Neuromorphic.Training;
using Xunit;
using Xunit.Abstractions;

namespace _103KNN.Neuromorphic.Tests;

public class BaselineBehaviorTests(ITestOutputHelper output)
{
    private static NeuromorphicLayerMap[] CreateXorTopology() =>
    [
        new NeuromorphicLayerMap(2, 8),
        new NeuromorphicLayerMap(8, 1)
    ];

    [Fact]
    public void IdenticalSeeds_ProduceIdenticalInitialPredictions()
    {
        var net1 = new NeuromorphicTrainableNetwork(CreateXorTopology(), seed: 103);
        var net2 = new NeuromorphicTrainableNetwork(CreateXorTopology(), seed: 103);

        float[] input = [0.25f, 0.75f];
        float[] p1 = net1.Predict(input);
        float[] p2 = net2.Predict(input);

        Assert.Equal(p1.Length, p2.Length);
        for (int i = 0; i < p1.Length; i++)
            Assert.Equal(p1[i], p2[i], precision: 6);
    }

    [Fact]
    public void TrainStep_ReducesMse_OnXorDataset()
    {
        var net = new NeuromorphicTrainableNetwork(CreateXorTopology(), seed: 103);
        float[][] inputs = [[0f, 0f], [0f, 1f], [1f, 0f], [1f, 1f]];
        float[][] targets = [[0f], [1f], [1f], [0f]];

        float initialMse = net.TrainStep(inputs, targets, learningRate: 0.01f);
        float finalMse = initialMse;
        for (int epoch = 0; epoch < 300; epoch++)
        {
            finalMse = net.TrainStep(inputs, targets, learningRate: 0.01f);
        }

        Assert.True(finalMse < initialMse, $"Expected MSE to decrease: initial={initialMse}, final={finalMse}");
        Assert.True(finalMse < 0.05f, $"Expected XOR MSE < 0.05 after 300 epochs, got {finalMse}");
        Assert.True(net.LastSpikesProcessed >= 0);
    }

    [Fact]
    public void Persistence_SaveAndLoad_PreservesPredictionsAndMinMaxScales()
    {
        var net = new NeuromorphicTrainableNetwork(CreateXorTopology(), seed: 103);
        float[][] inputs = [[0f, 0f], [0f, 1f], [1f, 0f], [1f, 1f]];
        float[][] targets = [[0f], [1f], [1f], [0f]];
        for (int i = 0; i < 40; i++)
            net.TrainStep(inputs, targets, learningRate: 0.01f);

        float[] minIn = [10f, 100f];
        float[] maxIn = [20f, 200f];
        float[] minOut = [5f];
        float[] maxOut = [15f];
        net.MinIn = minIn;
        net.MaxIn = maxIn;
        net.MinOut = minOut;
        net.MaxOut = maxOut;

        string tempFile = Path.Combine(Path.GetTempPath(), $"103knn_test_{Guid.NewGuid():N}.json");
        try
        {
            NeuromorphicModelPersistence.Save(net, tempFile, minIn, maxIn, minOut, maxOut);
            var loaded = NeuromorphicModelPersistence.Load(tempFile);

            float[] rawInput = [15f, 150f];
            float[] expected = net.PredictRaw(rawInput);
            float[] actual = loaded.PredictRaw(rawInput);

            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], precision: 5);
        }
        finally
        {
            if (File.Exists(tempFile)) File.Delete(tempFile);
        }
    }

    [Fact]
    public void NumericalReference_MatchesMasterBaseline()
    {
        var net = new NeuromorphicTrainableNetwork(CreateXorTopology(), seed: 103);
        float initPred = net.Predict([0.25f, 0.75f])[0];
        long initSpikes = net.LastSpikesProcessed;

        float[][] inputs = [[0f, 0f], [0f, 1f], [1f, 0f], [1f, 1f]];
        float[][] targets = [[0f], [1f], [1f], [0f]];

        float mse50 = 0f;
        for (int epoch = 0; epoch < 50; epoch++)
            mse50 = net.TrainStep(inputs, targets, learningRate: 0.01f);
        long trainSpikes = net.LastSpikesProcessed;
        float finalPred = net.Predict([1f, 0f])[0];

        output.WriteLine($"REF: initPred={initPred:R} initSpikes={initSpikes} mse50={mse50:R} trainSpikes={trainSpikes} finalPred={finalPred:R}");
        Assert.Equal(0.4690226f, initPred, precision: 6);
        Assert.Equal(3L, initSpikes);
        Assert.Equal(0.15155157f, mse50, precision: 6);
        Assert.Equal(4L, trainSpikes);
        Assert.Equal(0.6619006f, finalPred, precision: 6);
    }
}
