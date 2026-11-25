using System.Text.Json;
using System.IO;
using TorchSharp;
using NowLib;
using System.Collections.Generic;

namespace LibQuid;

public sealed class LibQuidLoadedModel
{
    public LibQuidNetwork Network { get; init; }
    public List<ConnectionMap> Maps { get; init; }
}

public static class LibQuidPersistence
{
    private const int FORMAT_VERSION = 2;

    private sealed class LibQuidModelPackage
    {
        public int FormatVersion { get; set; }
        public List<ConnectionMap> Maps { get; set; }
        public float DefaultDeltaTime { get; set; }
        public string TopologyHash { get; set; }
    }

    public static void SaveFullModel(LibQuidNetwork network, List<ConnectionMap> maps, string baseFilePath)
    {
        var package = new LibQuidModelPackage
        {
            FormatVersion = FORMAT_VERSION,
            Maps = maps,
            DefaultDeltaTime = network.DeltaTime,
            TopologyHash = network.ComputeTopologyHash(maps)
        };

        File.WriteAllText(baseFilePath + ".json", JsonSerializer.Serialize(package, new JsonSerializerOptions { WriteIndented = true }));
        network.save(baseFilePath + ".weights.bin");
    }

    public static LibQuidLoadedModel LoadFullModel(string baseFilePath)
    {
        string jsonPath = baseFilePath + ".json";
        string weightsPath = baseFilePath + ".weights.bin";

        if (!File.Exists(jsonPath)) throw new FileNotFoundException("Configuración no encontrada.");

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var package = JsonSerializer.Deserialize<LibQuidModelPackage>(File.ReadAllText(jsonPath), options);

        if (package == null || package.FormatVersion != FORMAT_VERSION)
            throw new System.Exception("Versión de modelo LibQuid no compatible.");

        var network = new LibQuidNetwork(package.Maps)
        {
            DeltaTime = package.DefaultDeltaTime
        };

        if (File.Exists(weightsPath))
        {
            network.load(weightsPath);
        }

        return new LibQuidLoadedModel { Network = network, Maps = package.Maps };
    }
}