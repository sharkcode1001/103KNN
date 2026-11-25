using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using NowLib;
using TorchSharp;

namespace NewNow;

public sealed class NewNowLoadedModel
{
    public NewNowNetwork Network { get; init; }
    public List<ConnectionMap> Maps { get; init; }
    public IActivation Activation { get; init; }
}

public static class NewNowPersistence
{
    private const int FORMAT_VERSION = 1;

    private sealed class NewNowModelPackage
    {
        public int FormatVersion { get; set; } = FORMAT_VERSION;
        public ActivationType Activation { get; set; } = ActivationType.ReLU;
        public List<ConnectionMap> Maps { get; set; } = new();
    }

    public static void SaveFullModel(NewNowNetwork network, List<ConnectionMap> maps, string baseFilePath)
    {
        var package = new NewNowModelPackage
        {
            Activation = network.Activation switch
            {
                ReLUActivation => ActivationType.ReLU,
                LeakyReLUActivation => ActivationType.LeakyReLU,
                SigmoidActivation => ActivationType.Sigmoid,
                _ => ActivationType.ReLU
            },
            Maps = maps
        };

        File.WriteAllText(baseFilePath + ".json", JsonSerializer.Serialize(package, new JsonSerializerOptions { WriteIndented = true }));
        network.SaveWeights(baseFilePath + ".weights.bin");
    }

    public static NewNowLoadedModel LoadFullModel(string baseFilePath)
    {
        string jsonPath = baseFilePath.EndsWith(".json") ? baseFilePath : baseFilePath + ".json";
        string weightsPath = jsonPath.Replace(".json", ".weights.bin");

        if (!File.Exists(jsonPath))
            throw new FileNotFoundException("Archivo JSON no encontrado.", jsonPath);

        var package = JsonSerializer.Deserialize<NewNowModelPackage>(File.ReadAllText(jsonPath))
            ?? throw new InvalidOperationException("No se pudo deserializar el modelo.");

        if (package.Maps == null || package.Maps.Count == 0)
            throw new InvalidOperationException("El modelo no contiene mapas de conexión.");

        IActivation activation = package.Activation switch
        {
            ActivationType.ReLU => new ReLUActivation(),
            ActivationType.LeakyReLU => new LeakyReLUActivation(),
            ActivationType.Sigmoid => new SigmoidActivation(),
            _ => new ReLUActivation()
        };

        var network = new NewNowNetwork(package.Maps, activation);

        if (File.Exists(weightsPath))
        {
            network.LoadWeights(weightsPath);
        }

        // Validación mínima
        if (network.Maps.Count != package.Maps.Count)
            throw new InvalidOperationException("Inconsistencia entre mapas cargados y estructura del modelo.");

        return new NewNowLoadedModel
        {
            Network = network,
            Maps = package.Maps,
            Activation = activation
        };
    }
}