using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using NowLib;

namespace NewNow;

public class NewNowNetwork : nn.Module<Tensor, Tensor>
{
    private readonly Sequential _layers;
    public IActivation Activation { get; }
    public IReadOnlyList<ConnectionMap> Maps { get; }

    public NewNowNetwork(List<ConnectionMap> maps, IActivation activation, bool useBatchNorm = false, bool trackRunningStats = true)
        : base(nameof(NewNowNetwork))
    {
        Activation = activation;
        Maps = maps.AsReadOnly();

        var modules = new List<(string name, nn.Module<Tensor, Tensor> module)>();

        for (int i = 0; i < maps.Count; i++)
        {
            var layer = new NewNowLayer(maps[i], activation, useBatchNorm, trackRunningStats);
            modules.Add(($"layer_{i}", layer));
        }

        _layers = nn.Sequential(modules);
        register_module("layers", _layers);
    }

    public override Tensor forward(Tensor input) => _layers.forward(input);

    /// <summary>
    /// Ejemplo de paso de entrenamiento. El modo train/eval debe manejarse fuera.
    /// </summary>
    public float TrainStep(Tensor input, Tensor target, optim.Optimizer optimizer)
    {
        optimizer.zero_grad();

        using (var _ = torch.NewDisposeScope())
        {
            var pred = forward(input);
            var loss = nn.functional.mse_loss(pred, target);

            loss.backward();

            // Enmascarar gradientes antes del step (ayuda, pero no suficiente solo)
            ApplyMasks(applyToWeights: false);

            optimizer.step();

            // Enmascarar PESOS después del step → clave para momentum/Adam/AdamW
            ApplyMasks(applyToWeights: true);

            return loss.item<float>();
        }
    }

    private void ApplyMasks(bool applyToWeights)
    {
        foreach (var module in modules())
        {
            if (module is NewNowLayer layer)
            {
                layer.ApplyTopologyMask(applyToWeights);
            }
        }
    }

    public void SaveWeights(string path) => this.save(path);
    public void LoadWeights(string path) => this.load(path);
}