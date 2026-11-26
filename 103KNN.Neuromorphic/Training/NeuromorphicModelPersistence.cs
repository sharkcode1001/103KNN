using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Training;

public static class NeuromorphicModelPersistence
{
    private const int FormatVersion = 1;

    private sealed class SerializedConnection
    {
        public int Source { get; set; }
        public int Target { get; set; }
    }

    private sealed class SerializedLayer
    {
        public int SourceSize { get; set; }
        public int TargetSize { get; set; }
        public List<SerializedConnection> Connections { get; set; } = new();
        public float[] Weights { get; set; } = Array.Empty<float>();
        public float[] Biases { get; set; } = Array.Empty<float>();
    }

    private sealed class NeuromorphicModelPackage
    {
        public int FormatVersion { get; set; } = 1;
        public CognitiveConfig Config { get; set; } = new();
        public List<SerializedLayer> Layers { get; set; } = new();
        public float[]? MinIn { get; set; }
        public float[]? MaxIn { get; set; }
        public float[]? MinOut { get; set; }
        public float[]? MaxOut { get; set; }
    }

    public static void Save(
        NeuromorphicTrainableNetwork network,
        string baseFilePath,
        float[]? minIn = null,
        float[]? maxIn = null,
        float[]? minOut = null,
        float[]? maxOut = null)
    {
        ArgumentNullException.ThrowIfNull(network);
        NeuromorphicModelPackage pkg = null!;
        network.ExecuteWithReadLock(() =>
        {
            pkg = new NeuromorphicModelPackage
            {
                FormatVersion = FormatVersion,
                Config = network.Config,
                MinIn = minIn ?? (float[]?)network.MinIn?.Clone(),
                MaxIn = maxIn ?? (float[]?)network.MaxIn?.Clone(),
                MinOut = minOut ?? (float[]?)network.MinOut?.Clone(),
                MaxOut = maxOut ?? (float[]?)network.MaxOut?.Clone()
            };

            for (int l = 0; l < network.Layers.Count; l++)
            {
                var layer = network.Layers[l];
                var map = network.Maps[l];

                var sLayer = new SerializedLayer
                {
                    SourceSize = layer.InputDim,
                    TargetSize = layer.OutputDim,
                    Biases = (float[])layer.Biases.Clone(),
                    Weights = (float[])layer.Weights.Clone()
                };

                foreach (var (src, dst) in map.Connections)
                {
                    sLayer.Connections.Add(new SerializedConnection { Source = src, Target = dst });
                }

                pkg.Layers.Add(sLayer);
            }
        });

        string jsonPath = baseFilePath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? baseFilePath
            : baseFilePath + ".json";

        var json = JsonSerializer.Serialize(pkg, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(jsonPath, json);
    }

    public static NeuromorphicTrainableNetwork Load(string baseFilePath)
    {
        string jsonPath = baseFilePath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? baseFilePath
            : baseFilePath + ".json";

        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("Archivo de modelo neuromórfico no encontrado.", jsonPath);

        var pkg = JsonSerializer.Deserialize<NeuromorphicModelPackage>(
            File.ReadAllText(jsonPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidOperationException("No se pudo deserializar el modelo neuromórfico.");

        var maps = new List<NeuromorphicLayerMap>();
        foreach (var sl in pkg.Layers)
        {
            var conns = new List<(int, int)>(sl.Connections.Count);
            foreach (var c in sl.Connections)
                conns.Add((c.Source, c.Target));

            maps.Add(new NeuromorphicLayerMap(sl.SourceSize, sl.TargetSize, conns));
        }

        var network = new NeuromorphicTrainableNetwork(maps, pkg.Config);

        for (int l = 0; l < network.Layers.Count; l++)
        {
            var layer = network.Layers[l];
            var sl = pkg.Layers[l];

            Array.Copy(sl.Biases, layer.Biases, Math.Min(sl.Biases.Length, layer.Biases.Length));
            Array.Copy(sl.Weights, layer.Weights, Math.Min(sl.Weights.Length, layer.Weights.Length));
        }

        network.MinIn = pkg.MinIn;
        network.MaxIn = pkg.MaxIn;
        network.MinOut = pkg.MinOut;
        network.MaxOut = pkg.MaxOut;

        return network;
    }
}
