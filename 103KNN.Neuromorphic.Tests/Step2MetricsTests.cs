using System;
using System.Threading.Tasks;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Training;
using Xunit;

namespace _103KNN.Neuromorphic.Tests;

public class Step2MetricsTests
{
    [Fact]
    public void SeparatedMetrics_PreserveLastSpikesProcessed_AndTrackExactCounts()
    {
        var net = new NeuromorphicTrainableNetwork(
        [
            new NeuromorphicLayerMap(2, 8),
            new NeuromorphicLayerMap(8, 1)
        ], seed: 103);

        float[] pred = net.Predict([0.25f, 0.75f], out var metrics);

        Assert.Single(pred);
        Assert.Equal(net.LastSpikesProcessed, 2 + metrics.LifFiredCount);
        Assert.Equal(metrics.LifFiredCount, net.LastLifFiredCount);
        Assert.Equal(metrics.PropagatedEventCount, net.LastPropagatedEventCount);
        Assert.Equal(metrics.SynapticOpsCount, net.LastSynapticOpsCount);

        // In Hybrid mode, propagated events >= input spikes (2) + sub-threshold activations
        Assert.True(metrics.PropagatedEventCount >= 2);
        // Synaptic ops must equal 2*8 (input->hidden) + hidden_propagated*1 (hidden->output)
        long hiddenPropagated = metrics.PropagatedEventCount - 2;
        Assert.Equal(2 * 8 + hiddenPropagated * 1, metrics.SynapticOpsCount);
    }

    [Fact]
    public void SparseTopology_HasFewerSynapticOps_ThanDenseTopology()
    {
        var dense = new NeuromorphicTrainableNetwork(
        [
            new NeuromorphicLayerMap(4, 8),
            new NeuromorphicLayerMap(8, 2)
        ], seed: 103);

        var sparse = new NeuromorphicTrainableNetwork(
        [
            new NeuromorphicLayerMap(4, 8, [(0, 0), (1, 1), (2, 2), (3, 3)]),
            new NeuromorphicLayerMap(8, 2, [(0, 0), (1, 1), (2, 0), (3, 1)])
        ], seed: 103);

        float[] input = [0.8f, 0.4f, 0.2f, 0.9f];
        dense.Predict(input, out var denseMetrics);
        sparse.Predict(input, out var sparseMetrics);

        Assert.True(sparseMetrics.SynapticOpsCount < denseMetrics.SynapticOpsCount,
            $"Expected sparse ops ({sparseMetrics.SynapticOpsCount}) < dense ops ({denseMetrics.SynapticOpsCount})");
    }

    [Fact]
    public void PredictRaw_SharesMeasurementMechanism_WithPredict()
    {
        var net = new NeuromorphicTrainableNetwork(
        [
            new NeuromorphicLayerMap(2, 6),
            new NeuromorphicLayerMap(6, 1)
        ], seed: 103)
        {
            MinIn = [10f, 100f],
            MaxIn = [20f, 200f],
            MinOut = [0f],
            MaxOut = [10f]
        };

        net.Predict([0.5f, 0.5f], out var normMetrics);
        net.PredictRaw([15f, 150f], out var rawMetrics);

        Assert.Equal(normMetrics, rawMetrics);
    }

    [Fact]
    public async Task ConcurrentPredictions_WithDifferentInputs_ReturnIndependentPerCallMetrics()
    {
        // Sparse network where input 0 connects to 4 hidden neurons, and input 1 connects to 1 hidden neuron
        var net = new NeuromorphicTrainableNetwork(
        [
            new NeuromorphicLayerMap(2, 5, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4)]),
            new NeuromorphicLayerMap(5, 1)
        ],
        new CognitiveConfig { BaseThreshold = 0.01f },
        seed: 103);

        // Zero-out input[1] or input[0] with different magnitudes/signs so each input produces distinct metrics
        float[] inputA = [1.0f, 0.0f];
        float[] inputB = [0.0f, 1.0f];

        net.Predict(inputA, out var expectedA);
        net.Predict(inputB, out var expectedB);

        var tasks = new Task[8];
        for (int t = 0; t < tasks.Length; t++)
        {
            bool useA = (t % 2 == 0);
            tasks[t] = Task.Run(() =>
            {
                for (int i = 0; i < 300; i++)
                {
                    if (useA)
                    {
                        net.Predict(inputA, out var mA);
                        Assert.Equal(expectedA, mA);
                    }
                    else
                    {
                        net.Predict(inputB, out var mB);
                        Assert.Equal(expectedB, mB);
                    }
                }
            });
        }

        await Task.WhenAll(tasks);
    }
}
