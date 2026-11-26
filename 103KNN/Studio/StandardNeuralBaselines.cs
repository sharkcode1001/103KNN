using System;
using System.Collections.Generic;
using System.Linq;
using NowLib;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace _103KNN.Studio;

/// <summary>
/// MLP estándar de TorchSharp usando capas nativas nn.Linear (sin máscaras de topología).
/// Sirve como referente directo de una red feedforward estándar de PyTorch.
/// </summary>
public sealed class StandardMlpNetwork : nn.Module<Tensor, Tensor>
{
    private readonly Sequential _seq;

    public StandardMlpNetwork(IReadOnlyList<int> layerSizes, IActivation activation) : base(nameof(StandardMlpNetwork))
    {
        var modules = new List<(string, nn.Module<Tensor, Tensor>)>();
        for (int i = 0; i < layerSizes.Count - 1; i++)
        {
            modules.Add(($"linear_{i}", nn.Linear(layerSizes[i], layerSizes[i + 1])));
            if (i < layerSizes.Count - 2)
            {
                modules.Add(($"act_{i}", new ActivationModuleWrapper(activation)));
            }
        }
        _seq = nn.Sequential(modules);
        register_module("seq", _seq);
    }

    public override Tensor forward(Tensor input) => _seq.forward(input);

    public float TrainStep(Tensor input, Tensor target, optim.Optimizer optimizer)
    {
        train();
        optimizer.zero_grad();
        using var pred = forward(input);
        using var loss = nn.functional.mse_loss(pred, target);
        loss.backward();
        optimizer.step();
        return loss.item<float>();
    }
}

/// <summary>
/// Arquitectura ResNet-Tabular (Gorishniy et al., NeurIPS 2021):
/// Proyección de entrada + bloques residuales con conexiones skip + cabeza de salida lineal.
/// </summary>
public sealed class ResNetTabularNetwork : nn.Module<Tensor, Tensor>
{
    private readonly Linear _inputProj;
    private readonly List<(Linear L1, Linear L2)> _blocks = new();
    private readonly Linear _head;

    public ResNetTabularNetwork(int inputDim, int hiddenWidth, int outputDim, int numBlocks = 2)
        : base(nameof(ResNetTabularNetwork))
    {
        int width = Math.Max(8, hiddenWidth);
        _inputProj = nn.Linear(inputDim, width);
        register_module("in_proj", _inputProj);

        for (int b = 0; b < numBlocks; b++)
        {
            var l1 = nn.Linear(width, width);
            var l2 = nn.Linear(width, width);
            _blocks.Add((l1, l2));
            register_module($"block_{b}_l1", l1);
            register_module($"block_{b}_l2", l2);
        }

        _head = nn.Linear(width, outputDim);
        register_module("head", _head);
    }

    public override Tensor forward(Tensor input)
    {
        using var h0 = _inputProj.forward(input);
        Tensor h = nn.functional.relu(h0);

        foreach (var (l1, l2) in _blocks)
        {
            using var z1 = l1.forward(h);
            using var a1 = nn.functional.relu(z1);
            using var z2 = l2.forward(a1);
            var next = h.add(z2);
            h.Dispose();
            h = nn.functional.relu(next);
            next.Dispose();
        }

        var outTensor = _head.forward(h);
        h.Dispose();
        return outTensor;
    }

    public float TrainStep(Tensor input, Tensor target, optim.Optimizer optimizer)
    {
        train();
        optimizer.zero_grad();
        using var pred = forward(input);
        using var loss = nn.functional.mse_loss(pred, target);
        loss.backward();
        optimizer.step();
        return loss.item<float>();
    }
}

/// <summary>
/// Red recurrente LSTM estándar de TorchSharp para series temporales (ventana de 5 pasos) o secuencias.
/// </summary>
public sealed class LstmTabularNetwork : nn.Module<Tensor, Tensor>
{
    private readonly LSTM _lstm;
    private readonly Linear _head;

    public LstmTabularNetwork(int inputDim, int hiddenDim, int outputDim) : base(nameof(LstmTabularNetwork))
    {
        int h = Math.Max(8, hiddenDim);
        _lstm = nn.LSTM(inputDim, h, numLayers: 1, batchFirst: true);
        _head = nn.Linear(h, outputDim);
        register_module("lstm", _lstm);
        register_module("head", _head);
    }

    public override Tensor forward(Tensor input)
    {
        var (output, _, _) = _lstm.forward(input);
        using (output)
        {
            long seqLen = output.shape[1];
            using var lastStep = output.select(1, seqLen - 1);
            return _head.forward(lastStep);
        }
    }

    public float TrainStep(Tensor input, Tensor target, optim.Optimizer optimizer)
    {
        train();
        optimizer.zero_grad();
        using var pred = forward(input);
        using var loss = nn.functional.mse_loss(pred, target);
        loss.backward();
        optimizer.step();
        return loss.item<float>();
    }
}

internal sealed class ActivationModuleWrapper : nn.Module<Tensor, Tensor>
{
    private readonly IActivation _activation;
    public ActivationModuleWrapper(IActivation activation) : base(nameof(ActivationModuleWrapper))
        => _activation = activation;
    public override Tensor forward(Tensor input) => _activation.Activate(input);
}
