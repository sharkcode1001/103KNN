using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Neurons;
using _103KNN.Neuromorphic.Training;
using Xunit;
using Xunit.Abstractions;

namespace _103KNN.Neuromorphic.Tests;

public class Step5_2CorrectnessTests(ITestOutputHelper output)
{
    [Fact]
    public void ExactZeroInputs_AreSkipped_WhileTinyNonZeroSignalsRemainActive()
    {
        var net = new NeuromorphicTrainableNetwork(
        [
            new NeuromorphicLayerMap(4, 6),
            new NeuromorphicLayerMap(6, 1)
        ],
        new CognitiveConfig { ExecutionMode = NeuromorphicExecutionMode.StrictSpiking },
        seed: 103);

        // Zero out biases so hidden layer cannot fire on bias alone
        Array.Clear(net.Layers[0].Biases);
        Array.Clear(net.Layers[1].Biases);

        net.Predict([0f, 0f, 0f, 0f], out var zeroMetrics);
        Assert.Equal(0L, zeroMetrics.LifFiredCount);
        Assert.Equal(0L, zeroMetrics.PropagatedEventCount);
        Assert.Equal(0L, zeroMetrics.SynapticOpsCount);

        // Small non-zero input (1e-5f) must NOT be discarded
        net.Predict([1e-5f, 0f, 0f, 0f], out var tinyMetrics);
        Assert.True(tinyMetrics.PropagatedEventCount >= 1);
        Assert.True(tinyMetrics.SynapticOpsCount >= 6);
    }

    [Fact]
    public void CognitiveLifNeuron_MatchesCanonicalTimeDecay_IncludingDtZero()
    {
        var cfg = new CognitiveConfig
        {
            BaseThreshold = 0.5f,
            MembraneDecay = 0.8f,
            ThresholdDecay = 0.9f,
            ThresholdAdaptationRate = 0.1f,
            TimeWindow = 1.0f
        };

        var neuron = new CognitiveLifNeuron(0, cfg);

        // dt == 0 at timestamp 0.0 -> leak factor is exactly 1.0
        bool fired0 = neuron.Integrate(0.2f, 0.0f);
        Assert.False(fired0);
        Assert.Equal(0.2f, neuron.Voltage, precision: 6);

        // Simultaneous second spike at timestamp 0.0 (dt == 0) -> no decay
        bool firedSameT = neuron.Integrate(0.1f, 0.0f);
        Assert.False(firedSameT);
        Assert.Equal(0.3f, neuron.Voltage, precision: 6);

        // Spike at timestamp 0.5 -> dt = 0.5, expected voltage = 0.3 * 0.8^0.5 + 0.1
        bool firedHalf = neuron.Integrate(0.1f, 0.5f);
        float expectedV = 0.3f * MathF.Pow(0.8f, 0.5f) + 0.1f;
        Assert.False(firedHalf);
        Assert.Equal(expectedV, neuron.Voltage, precision: 5);

        // Firing spike at timestamp 0.75 -> triggers reset and threshold adaptation
        bool firedTrigger = neuron.Integrate(0.4f, 0.75f);
        Assert.True(firedTrigger);
        Assert.Equal(0.0f, neuron.Voltage, precision: 6);
        Assert.Equal(0.75f, neuron.FirstSpikeTime, precision: 6);
        Assert.Equal(0.6f, neuron.Threshold, precision: 5);
    }

    [Fact]
    public void SaveCheckpoint_AndLoad_ResumesAdamIdenticallyToUninterruptedTraining()
    {
        NeuromorphicLayerMap[] topo() => [new NeuromorphicLayerMap(2, 8), new NeuromorphicLayerMap(8, 1)];
        float[][] inputs = [[0f, 0f], [0f, 1f], [1f, 0f], [1f, 1f]];
        float[][] targets = [[0f], [1f], [1f], [0f]];

        var uninterrupted = new NeuromorphicTrainableNetwork(topo(), seed: 103);
        float mseUninterrupted = 0f;
        for (int i = 0; i < 50; i++)
            mseUninterrupted = uninterrupted.TrainStep(inputs, targets, learningRate: 0.01f);

        var splitNet = new NeuromorphicTrainableNetwork(topo(), seed: 103);
        for (int i = 0; i < 25; i++)
            splitNet.TrainStep(inputs, targets, learningRate: 0.01f);

        string ckptPath = Path.Combine(Path.GetTempPath(), $"103knn_ckpt_{Guid.NewGuid():N}.json");
        try
        {
            NeuromorphicModelPersistence.SaveCheckpoint(splitNet, ckptPath);
            var resumed = NeuromorphicModelPersistence.Load(ckptPath);

            float mseResumed = 0f;
            for (int i = 0; i < 25; i++)
                mseResumed = resumed.TrainStep(inputs, targets, learningRate: 0.01f);

            Assert.Equal(mseUninterrupted, mseResumed, precision: 6);
            for (int l = 0; l < uninterrupted.Layers.Count; l++)
            {
                for (int w = 0; w < uninterrupted.Layers[l].Weights.Length; w++)
                    Assert.Equal(uninterrupted.Layers[l].Weights[w], resumed.Layers[l].Weights[w], precision: 5);
            }
        }
        finally
        {
            if (File.Exists(ckptPath)) File.Delete(ckptPath);
        }
    }

    [Fact]
    public void Load_RejectsCorruptedOrMismatchedModelFiles()
    {
        string badJsonPath = Path.Combine(Path.GetTempPath(), $"103knn_bad_{Guid.NewGuid():N}.json");
        try
        {
            // Mismatched Weights length (expected 2*4=8, provided 2)
            File.WriteAllText(badJsonPath, """
            {
              "FormatVersion": 1,
              "Config": { "BaseThreshold": 0.25, "MembraneDecay": 0.9, "LateralInhibitionFactor": 0.05, "ThresholdAdaptationRate": 0.02, "TimeWindow": 1.0 },
              "Layers": [
                { "SourceSize": 2, "TargetSize": 4, "Connections": [], "Weights": [0.1, 0.2], "Biases": [0, 0, 0, 0] }
              ]
            }
            """);
            Assert.Throws<InvalidDataException>(() => NeuromorphicModelPersistence.Load(badJsonPath));

            // Unsupported FormatVersion
            File.WriteAllText(badJsonPath, """
            {
              "FormatVersion": 99,
              "Layers": [
                { "SourceSize": 1, "TargetSize": 1, "Connections": [], "Weights": [0.1], "Biases": [0.0] }
              ]
            }
            """);
            Assert.Throws<InvalidDataException>(() => NeuromorphicModelPersistence.Load(badJsonPath));
        }
        finally
        {
            if (File.Exists(badJsonPath)) File.Delete(badJsonPath);
        }
    }

    [Fact]
    public void FiveSeedStatisticalBenchmark_WithRepeatedLatencyMeasurements_IsFiniteAndConsistent()
    {
        int[] seeds = [103, 202, 307, 409, 521];
        float[][] inputs = [[0f, 0f], [0f, 1f], [1f, 0f], [1f, 1f]];
        float[][] targets = [[0f], [1f], [1f], [0f]];

        var mses = new List<double>(seeds.Length);
        var latenciesUs = new List<double>(seeds.Length * 50);

        foreach (int seed in seeds)
        {
            var net = new NeuromorphicTrainableNetwork(
            [
                new NeuromorphicLayerMap(2, 8),
                new NeuromorphicLayerMap(8, 1)
            ], seed: seed);

            float mse = 0f;
            for (int epoch = 0; epoch < 200; epoch++)
                mse = net.TrainStep(inputs, targets, learningRate: 0.01f);

            Assert.True(float.IsFinite(mse));
            mses.Add(mse);

            // Warmup
            for (int w = 0; w < 10; w++) net.Predict([0.5f, 0.5f]);

            for (int r = 0; r < 50; r++)
            {
                long t0 = Stopwatch.GetTimestamp();
                net.Predict([0.5f, 0.5f]);
                long t1 = Stopwatch.GetTimestamp();
                latenciesUs.Add((t1 - t0) * 1_000_000.0 / Stopwatch.Frequency);
            }
        }

        double meanMse = mses.Average();
        double stdMse = Math.Sqrt(mses.Average(m => Math.Pow(m - meanMse, 2)));
        latenciesUs.Sort();
        double p50Us = latenciesUs[latenciesUs.Count / 2];
        double p95Us = latenciesUs[(int)(latenciesUs.Count * 0.95)];

        output.WriteLine($"5-Seed MSE: {meanMse:F6} ± {stdMse:F6} | Latency p50={p50Us:F2} us, p95={p95Us:F2} us");
        Assert.True(meanMse < 0.1);
        Assert.True(p50Us > 0 && p95Us >= p50Us);
    }
}
