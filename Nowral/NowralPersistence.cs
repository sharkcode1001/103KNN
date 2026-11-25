using System.Text.Json;
using System.Security.Cryptography;
using System.Text;
using NowLib;

namespace Nowral;

public sealed class NowralLoadedModel
{
    public NowralNetwork Network { get; init; }
    public List<ConnectionMap> Maps { get; init; }
    public IActivation Activation { get; init; }
}

public static class NowralPersistence
{
    private const int FORMAT_VERSION = 1;

    private sealed class NowralModelPackage
    {
        public int FormatVersion { get; set; }
        public string ActivationType { get; set; }
        public List<ConnectionMap> Maps { get; set; }
        public long ParameterCount { get; set; }
        public string TopologyHash { get; set; }
    }

    public static void SaveFullModel(NowralNetwork network, List<ConnectionMap> maps, string baseFilePath)
    {
        var package = new NowralModelPackage
        {
            FormatVersion = FORMAT_VERSION,
            ActivationType = network.Activation.GetType().Name,
            Maps = maps,
            ParameterCount = network.state_dict().Sum(p => p.Value.numel()),
            TopologyHash = ComputeTopologyHash(maps)
        };

        File.WriteAllText(baseFilePath + ".json", JsonSerializer.Serialize(package, new JsonSerializerOptions { WriteIndented = true }));
        network.SaveWeights(baseFilePath + ".weights.bin");
    }

    public static NowralLoadedModel LoadFullModel(string baseFilePath)
    {
        string jsonPath = baseFilePath + ".json";
        string weightsPath = baseFilePath + ".weights.bin";

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var package = JsonSerializer.Deserialize<NowralModelPackage>(File.ReadAllText(jsonPath), options);

        if (package == null || package.FormatVersion != FORMAT_VERSION)
            throw new Exception("Versión de modelo no compatible.");

        IActivation activation = package.ActivationType switch
        {
            "ReLUActivation" => new ReLUActivation(),
            "SigmoidActivation" => new SigmoidActivation(),
            _ => throw new Exception("Activación desconocida")
        };

        var network = new NowralNetwork(package.Maps, activation);
        network.LoadWeights(weightsPath);

        return new NowralLoadedModel { Network = network, Maps = package.Maps, Activation = activation };
    }

    private static string ComputeTopologyHash(List<ConnectionMap> maps)
    {
        var json = JsonSerializer.Serialize(maps);
        using var sha = SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(Encoding.UTF8.GetBytes(json)));
    }
}