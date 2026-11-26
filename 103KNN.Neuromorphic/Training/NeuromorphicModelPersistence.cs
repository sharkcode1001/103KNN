using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Training;

public static class NeuromorphicModelPersistence
{
    private const int InferenceFormatVersion = 1;
    private const int CheckpointFormatVersion = 2;

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
        public float[]? AdamM { get; set; }
        public float[]? AdamV { get; set; }
        public float[]? BiasM { get; set; }
        public float[]? BiasV { get; set; }
    }

    private sealed class NeuromorphicModelPackage
    {
        public int FormatVersion { get; set; } = InferenceFormatVersion;
        public int AdamStep { get; set; }
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
        => SaveInternal(network, baseFilePath, includeOptimizerState: false, minIn, maxIn, minOut, maxOut);

    /// <summary>
    /// Guarda un checkpoint completo de entrenamiento (incluyendo <c>AdamStep</c>, <c>AdamM</c>, <c>AdamV</c>, <c>BiasM</c> y <c>BiasV</c>)
    /// para reanudar entrenamiento sin perder el estado del optimizador.
    /// </summary>
    public static void SaveCheckpoint(
        NeuromorphicTrainableNetwork network,
        string baseFilePath,
        float[]? minIn = null,
        float[]? maxIn = null,
        float[]? minOut = null,
        float[]? maxOut = null)
        => SaveInternal(network, baseFilePath, includeOptimizerState: true, minIn, maxIn, minOut, maxOut);

    private static void SaveInternal(
        NeuromorphicTrainableNetwork network,
        string baseFilePath,
        bool includeOptimizerState,
        float[]? minIn,
        float[]? maxIn,
        float[]? minOut,
        float[]? maxOut)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseFilePath);

        NeuromorphicModelPackage pkg = null!;
        network.ExecuteWithReadLock(() =>
        {
            pkg = new NeuromorphicModelPackage
            {
                FormatVersion = includeOptimizerState ? CheckpointFormatVersion : InferenceFormatVersion,
                AdamStep = includeOptimizerState ? network.AdamStep : 0,
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

                if (includeOptimizerState)
                {
                    sLayer.AdamM = (float[])layer.AdamM.Clone();
                    sLayer.AdamV = (float[])layer.AdamV.Clone();
                    sLayer.BiasM = (float[])layer.BiasM.Clone();
                    sLayer.BiasV = (float[])layer.BiasV.Clone();
                }

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

    private static void ValidateFiniteArray(float[] arr, string fieldName)
    {
        for (int i = 0; i < arr.Length; i++)
        {
            if (!float.IsFinite(arr[i]))
                throw new InvalidDataException($"El campo '{fieldName}' contiene un valor no finito en el índice {i}.");
        }
    }

    public static NeuromorphicTrainableNetwork Load(string baseFilePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseFilePath);

        string jsonPath = baseFilePath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? baseFilePath
            : baseFilePath + ".json";

        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("Archivo de modelo neuromórfico no encontrado.", jsonPath);

        var pkg = JsonSerializer.Deserialize<NeuromorphicModelPackage>(
            File.ReadAllText(jsonPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidOperationException("No se pudo deserializar el modelo neuromórfico.");

        if (pkg.FormatVersion != InferenceFormatVersion && pkg.FormatVersion != CheckpointFormatVersion)
            throw new InvalidDataException($"Versión de formato no soportada: {pkg.FormatVersion}.");
        if (pkg.Layers == null || pkg.Layers.Count == 0)
            throw new InvalidDataException("El modelo deserializado no contiene capas.");
        if (pkg.AdamStep < 0)
            throw new InvalidDataException($"AdamStep inválido: {pkg.AdamStep}.");

        var maps = new List<NeuromorphicLayerMap>(pkg.Layers.Count);
        foreach (var sl in pkg.Layers)
        {
            if (sl == null)
                throw new InvalidDataException("El modelo contiene una capa nula.");
            var conns = new List<(int, int)>(sl.Connections?.Count ?? 0);
            if (sl.Connections != null)
            {
                foreach (var c in sl.Connections)
                    conns.Add((c.Source, c.Target));
            }

            maps.Add(new NeuromorphicLayerMap(sl.SourceSize, sl.TargetSize, conns));
        }

        var network = new NeuromorphicTrainableNetwork(maps, pkg.Config);

        for (int l = 0; l < network.Layers.Count; l++)
        {
            var layer = network.Layers[l];
            var sl = pkg.Layers[l];

            if (sl.Biases == null || sl.Biases.Length != layer.OutputDim)
                throw new InvalidDataException($"Capa {l}: longitud de Biases ({sl.Biases?.Length ?? 0}) no coincide con OutputDim ({layer.OutputDim}).");
            if (sl.Weights == null || sl.Weights.Length != layer.Weights.Length)
                throw new InvalidDataException($"Capa {l}: longitud de Weights ({sl.Weights?.Length ?? 0}) no coincide con {layer.Weights.Length}.");

            ValidateFiniteArray(sl.Biases, $"Layers[{l}].Biases");
            ValidateFiniteArray(sl.Weights, $"Layers[{l}].Weights");

            Array.Copy(sl.Biases, layer.Biases, layer.OutputDim);
            Array.Copy(sl.Weights, layer.Weights, layer.Weights.Length);

            if (pkg.FormatVersion == CheckpointFormatVersion)
            {
                if (sl.AdamM == null || sl.AdamM.Length != layer.AdamM.Length)
                    throw new InvalidDataException($"Capa {l}: longitud de AdamM inválida en checkpoint.");
                if (sl.AdamV == null || sl.AdamV.Length != layer.AdamV.Length)
                    throw new InvalidDataException($"Capa {l}: longitud de AdamV inválida en checkpoint.");
                if (sl.BiasM == null || sl.BiasM.Length != layer.BiasM.Length)
                    throw new InvalidDataException($"Capa {l}: longitud de BiasM inválida en checkpoint.");
                if (sl.BiasV == null || sl.BiasV.Length != layer.BiasV.Length)
                    throw new InvalidDataException($"Capa {l}: longitud de BiasV inválida en checkpoint.");

                ValidateFiniteArray(sl.AdamM, $"Layers[{l}].AdamM");
                ValidateFiniteArray(sl.AdamV, $"Layers[{l}].AdamV");
                ValidateFiniteArray(sl.BiasM, $"Layers[{l}].BiasM");
                ValidateFiniteArray(sl.BiasV, $"Layers[{l}].BiasV");

                Array.Copy(sl.AdamM, layer.AdamM, layer.AdamM.Length);
                Array.Copy(sl.AdamV, layer.AdamV, layer.AdamV.Length);
                Array.Copy(sl.BiasM, layer.BiasM, layer.BiasM.Length);
                Array.Copy(sl.BiasV, layer.BiasV, layer.BiasV.Length);
            }
        }

        int expectedIn = network.Layers[0].InputDim;
        int expectedOut = network.Layers[^1].OutputDim;
        if ((pkg.MinIn == null) != (pkg.MaxIn == null) ||
            (pkg.MinIn != null && (pkg.MinIn.Length != expectedIn || pkg.MaxIn!.Length != expectedIn)))
            throw new InvalidDataException($"Las escalas MinIn/MaxIn deben ser nulas o tener longitud {expectedIn}.");
        if ((pkg.MinOut == null) != (pkg.MaxOut == null) ||
            (pkg.MinOut != null && (pkg.MinOut.Length != expectedOut || pkg.MaxOut!.Length != expectedOut)))
            throw new InvalidDataException($"Las escalas MinOut/MaxOut deben ser nulas o tener longitud {expectedOut}.");

        if (pkg.MinIn != null)
        {
            ValidateFiniteArray(pkg.MinIn, nameof(pkg.MinIn));
            ValidateFiniteArray(pkg.MaxIn!, nameof(pkg.MaxIn));
        }
        if (pkg.MinOut != null)
        {
            ValidateFiniteArray(pkg.MinOut, nameof(pkg.MinOut));
            ValidateFiniteArray(pkg.MaxOut!, nameof(pkg.MaxOut));
        }

        network.AdamStep = pkg.FormatVersion == CheckpointFormatVersion ? pkg.AdamStep : 0;
        network.MinIn = pkg.MinIn;
        network.MaxIn = pkg.MaxIn;
        network.MinOut = pkg.MinOut;
        network.MaxOut = pkg.MaxOut;

        return network;
    }
}
