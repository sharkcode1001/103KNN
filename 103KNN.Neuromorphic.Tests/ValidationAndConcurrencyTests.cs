using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Training;
using Xunit;

namespace _103KNN.Neuromorphic.Tests;

public class ValidationAndConcurrencyTests
{
    private static NeuromorphicLayerMap[] ValidXorTopology() =>
    [
        new NeuromorphicLayerMap(2, 6),
        new NeuromorphicLayerMap(6, 1)
    ];

    [Fact]
    public void LayerMap_InvalidDimensionsOrIndices_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicLayerMap(0, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicLayerMap(4, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicLayerMap(2, 2, [(2, 0)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicLayerMap(2, 2, [(0, -1)]));
    }

    [Fact]
    public void Constructor_InvalidTopologyOrConfig_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new NeuromorphicTrainableNetwork(null!));
        Assert.Throws<ArgumentException>(() => new NeuromorphicTrainableNetwork([]));

        // Discontinuous layer dimensions: 2->4 followed by 5->1
        Assert.Throws<ArgumentException>(() => new NeuromorphicTrainableNetwork([
            new NeuromorphicLayerMap(2, 4),
            new NeuromorphicLayerMap(5, 1)
        ]));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { BaseThreshold = 0f }));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { MembraneDecay = 1.5f }));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { TimeWindow = float.NaN }));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { ThresholdDecay = float.PositiveInfinity }));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { MaxDeliberationSteps = 0 }));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { EarlyExitConfidenceThreshold = float.NaN }));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { LateralInhibitionFactor = float.NegativeInfinity }));

        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuromorphicTrainableNetwork(
            ValidXorTopology(),
            new CognitiveConfig { ThresholdAdaptationRate = float.NaN }));
    }

    [Fact]
    public void LayerMap_DuplicateConnections_AreDeduplicatedSafely()
    {
        var map = new NeuromorphicLayerMap(2, 2, [(0, 0), (0, 0), (1, 1)]);
        var net = new NeuromorphicTrainableNetwork([map, new NeuromorphicLayerMap(2, 1)]);
        Assert.Single(net.Layers[0].ConnectedTargets[0]);
        Assert.Single(net.Layers[0].ConnectedTargets[1]);
        Assert.False(net.Layers[0].IsFullyConnected);
    }

    [Fact]
    public void ExceptionDuringOperation_DoesNotDeadlockSubsequentCalls()
    {
        var net = new NeuromorphicTrainableNetwork(ValidXorTopology(), seed: 103);

        // Pre-lock validation exceptions
        Assert.Throws<ArgumentOutOfRangeException>(() => net.Predict([float.NaN, 0.5f]));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.TrainStep([[0f, float.NaN]], [[1f]]));

        // Exception thrown directly INSIDE the lock-protected section
        Assert.Throws<InvalidOperationException>(() =>
            net.ExecuteWithReadLock(() => throw new InvalidOperationException("Simulated failure inside read lock")));

        // Must be able to acquire WriteLock (TrainStep) and ReadLock (Predict) immediately without deadlock
        float mse = net.TrainStep([[0f, 1f]], [[1f]]);
        Assert.True(float.IsFinite(mse));

        float[] validPred = net.Predict([0.5f, 0.5f]);
        Assert.Single(validPred);
        Assert.True(float.IsFinite(validPred[0]));
    }

    [Fact]
    public async Task ParallelPredictions_ReturnIndependentValidResults()
    {
        var net = new NeuromorphicTrainableNetwork(ValidXorTopology(), seed: 103);
        float[] expectedA = net.Predict([0.1f, 0.9f]);
        float[] expectedB = net.Predict([0.9f, 0.1f]);

        var t1 = Task.Run(() => net.Predict([0.1f, 0.9f]));
        var t2 = Task.Run(() => net.Predict([0.9f, 0.1f]));
        float[][] results = await Task.WhenAll(t1, t2);

        Assert.Equal(expectedA[0], results[0][0], precision: 6);
        Assert.Equal(expectedB[0], results[1][0], precision: 6);
    }

    [Fact]
    public void Predict_InvalidLengthOrNonFinite_Throws()
    {
        var net = new NeuromorphicTrainableNetwork(ValidXorTopology());

        Assert.Throws<ArgumentException>(() => net.Predict([0.5f]));
        Assert.Throws<ArgumentException>(() => net.Predict([0.5f, 0.5f, 0.5f]));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.Predict([float.NaN, 0.5f]));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.Predict([0.5f, float.PositiveInfinity]));
    }

    [Fact]
    public void PredictRaw_InvalidLengthNonFiniteOrMismatchedScales_Throws()
    {
        var net = new NeuromorphicTrainableNetwork(ValidXorTopology());

        Assert.Throws<ArgumentException>(() => net.PredictRaw([1f]));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.PredictRaw([float.NegativeInfinity, 1f]));

        net.MinIn = [0f, 0f];
        Assert.Throws<InvalidOperationException>(() => net.PredictRaw([0.5f, 0.5f]));

        net.MaxIn = [1f]; // mismatched length
        Assert.Throws<InvalidOperationException>(() => net.PredictRaw([0.5f, 0.5f]));

        net.MaxIn = [1f, 1f];
        net.MinOut = [0f, 0f]; // OutputDim is 1, not 2
        net.MaxOut = [1f, 1f];
        Assert.Throws<InvalidOperationException>(() => net.PredictRaw([0.5f, 0.5f]));
    }

    [Fact]
    public void TrainStep_InvalidArguments_Throws()
    {
        var net = new NeuromorphicTrainableNetwork(ValidXorTopology());
        float[][] x = [[0f, 1f]];
        float[][] y = [[1f]];

        Assert.Throws<ArgumentNullException>(() => net.TrainStep(null!, y));
        Assert.Throws<ArgumentNullException>(() => net.TrainStep(x, null!));
        Assert.Throws<ArgumentException>(() => net.TrainStep(x, [[1f], [0f]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.TrainStep(x, y, learningRate: 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.TrainStep(x, y, learningRate: float.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.TrainStep(x, y, stdpRate: -0.1f));
        Assert.Throws<ArgumentException>(() => net.TrainStep([[0f]], y));
        Assert.Throws<ArgumentException>(() => net.TrainStep(x, [[0f, 1f]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.TrainStep([[float.NaN, 1f]], y));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.TrainStep(x, [[float.PositiveInfinity]]));
    }

    [Fact]
    public async Task ConcurrentTrainAndPredict_RemainsThreadSafeAndFinite()
    {
        var net = new NeuromorphicTrainableNetwork(ValidXorTopology(), seed: 103)
        {
            MinIn = [0f, 0f],
            MaxIn = [10f, 10f],
            MinOut = [0f],
            MaxOut = [100f]
        };

        float[][] inputs = [[0f, 0f], [0f, 1f], [1f, 0f], [1f, 1f]];
        float[][] targets = [[0f], [1f], [1f], [0f]];

        float[] initialWeights = (float[])net.Layers[0].Weights.Clone();
        var errors = new List<Exception>();

        Task trainer = Task.Run(() =>
        {
            try
            {
                for (int i = 0; i < 300; i++)
                {
                    float mse = net.TrainStep(inputs, targets, learningRate: 0.01f);
                    Assert.True(float.IsFinite(mse));
                }
            }
            catch (Exception ex)
            {
                lock (errors) errors.Add(ex);
            }
        });

        var readers = new Task[4];
        for (int r = 0; r < readers.Length; r++)
        {
            readers[r] = Task.Run(() =>
            {
                try
                {
                    for (int i = 0; i < 500; i++)
                    {
                        float[] p = net.Predict([0.5f, 0.5f]);
                        float[] pr = net.PredictRaw([5f, 5f]);
                        Assert.Single(p);
                        Assert.Single(pr);
                        Assert.True(float.IsFinite(p[0]));
                        Assert.True(float.IsFinite(pr[0]));
                        Assert.True(net.LastSpikesProcessed >= 0);
                    }
                }
                catch (Exception ex)
                {
                    lock (errors) errors.Add(ex);
                }
            });
        }

        await Task.WhenAll([trainer, .. readers]);
        Assert.Empty(errors);

        bool weightsChanged = false;
        for (int i = 0; i < initialWeights.Length; i++)
        {
            if (initialWeights[i] != net.Layers[0].Weights[i])
            {
                weightsChanged = true;
                break;
            }
        }
        Assert.True(weightsChanged, "Expected weights to be updated during concurrent training.");
    }
}
