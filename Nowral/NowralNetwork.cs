using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using NowLib;

namespace Nowral;

public class NowralNetwork : nn.Module<Tensor, Tensor>
{
    private readonly Sequential _layers;
    public IActivation Activation { get; }

    public NowralNetwork(List<ConnectionMap> maps, IActivation activation) : base(nameof(NowralNetwork))
    {
        Activation = activation;
        var modules = new List<(string, nn.Module<Tensor, Tensor>)>();
        for (int i = 0; i < maps.Count; i++)
            modules.Add(($"layer_{i}", new NowralLayer(maps[i], activation)));
        _layers = nn.Sequential(modules);
        register_module("layers", _layers);
    }

    public override Tensor forward(Tensor input) => _layers.forward(input);

    public float TrainStep(Tensor input, Tensor target, optim.Optimizer optimizer)
    {
        train();
        optimizer.zero_grad();
        using var pred = forward(input);
        using var loss = nn.functional.mse_loss(pred, target);
        loss.backward();
        foreach (var m in _layers.children()) if (m is NowralLayer l) l.ApplyGradientMask();
        optimizer.step();
        return loss.item<float>();
    }

    public void SaveWeights(string path) => this.save(path);
    public void LoadWeights(string path) => this.load(path);

    public static bool ValidateStructure(List<ConnectionMap> maps) => maps != null && maps.Count > 0;
}