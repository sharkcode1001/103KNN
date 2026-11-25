using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using NowLib;

namespace NewNow;

public class NewNowLayer : nn.Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly Parameter bias;
    private readonly BatchNorm1d? _batchNorm;
    private readonly IActivation _activation;
    private readonly string _maskBufferName = "topology_mask";

    public NewNowLayer(ConnectionMap map, IActivation activation, bool useBatchNorm = false, bool trackRunningStats = true)
        : base(nameof(NewNowLayer))
    {
        _activation = activation;

        var mask = map.GenerateMask();
        register_buffer(_maskBufferName, mask);

        // Inicialización Kaiming (He) normal
        var rawWeight = randn(new long[] { map.TargetSize, map.SourceSize }, dtype: ScalarType.Float32)
                        * (float)Math.Sqrt(2.0 / map.SourceSize);

        // Aplicar máscara desde el inicio → conexiones inexistentes = 0
        using (no_grad())
        {
            rawWeight.mul_(mask);
        }

        weight = nn.Parameter(rawWeight);

        // Bias en 0 → razonable para ReLU/LeakyReLU
        // Para Sigmoid a veces se prefiere ~0.5, pero lo dejamos configurable externamente si hace falta
        bias = nn.Parameter(zeros(new long[] { map.TargetSize }, dtype: ScalarType.Float32));

        if (useBatchNorm)
        {
            _batchNorm = nn.BatchNorm1d(map.TargetSize, track_running_stats: trackRunningStats);
            register_module("batchNorm", _batchNorm);
        }

        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }

    private Tensor Mask => get_buffer(_maskBufferName);

    public override Tensor forward(Tensor input)
    {
        var z = nn.functional.linear(input, weight, bias);

        if (_batchNorm is not null)
        {
            z = _batchNorm.forward(z);
        }

        return _activation.Activate(z);
    }

    /// <summary>
    /// Aplica la máscara de topología a pesos o gradientes.
    /// Llamar después del optimizer.step() con applyToWeights = true es la forma más robusta.
    /// </summary>
    public void ApplyTopologyMask(bool applyToWeights)
    {
        using (no_grad())
        {
            if (applyToWeights)
            {
                weight.mul_(Mask);           // ← más seguro con momentum/Adam
            }
            else if (weight.grad is { } grad)
            {
                grad.mul_(Mask);
            }
        }
    }
}