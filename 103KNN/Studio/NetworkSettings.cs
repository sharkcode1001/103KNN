using System;
using System.Collections.Generic;
using System.Linq;
using NowLib;

namespace _103KNN.Studio;

public enum NetworkArchitectureType
{
    Nowral = 0,
    LibQuid = 1,
    NewNow = 2,
    Neuromorphic = 3,
    FastTree = 4,
    FastForest = 5,
    StandardMlp = 6,
    ResNetTabular = 7,
    Lstm = 8
}

public enum OptimizerType
{
    Adam,
    SGD,
    RMSprop
}

public class GlobalNetworkSettings
{
    public string Structure { get; set; } = "8 32 16 1";
    public ActivationType Activation { get; set; } = ActivationType.LeakyReLU;

    public int InputCount => ParseStructure().FirstOrDefault();
    public int OutputCount => ParseStructure().LastOrDefault();
    public int[] HiddenLayers
    {
        get
        {
            var list = ParseStructure();
            return list.Count >= 2 ? list.Skip(1).Take(list.Count - 2).ToArray() : Array.Empty<int>();
        }
    }

    public List<int> ParseStructure()
    {
        return Structure.Split(new[] { ' ', '-', ',' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(s => int.TryParse(s.Trim(), out int val) ? Math.Max(1, val) : 0)
            .Where(v => v > 0)
            .ToList();
    }
}

public class TrainingSettings
{
    public double LearningRate { get; set; } = 0.005;
    public int Epochs { get; set; } = 100;
    public int BatchSize { get; set; } = 64;
    public OptimizerType Optimizer { get; set; } = OptimizerType.Adam;
    public double TargetError { get; set; } = 0.0001;
    public double StdpRate { get; set; } = 0.002;
}

public sealed class PredictionResultDto
{
    public bool Success { get; init; }
    public float[]? Prediction { get; init; }
    public string? Error { get; init; }

    public static PredictionResultDto Ok(float[] values)
        => new() { Success = true, Prediction = values };

    public static PredictionResultDto Fail(string error)
        => new() { Success = false, Error = error };
}

public sealed class TrainingProgressDto
{
    public long Epoch { get; init; }
    public float MSE { get; init; }
    public bool IsTraining { get; init; }
}
