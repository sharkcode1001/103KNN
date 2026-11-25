using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using NowLib;

namespace LibQuid;

public class LibQuidNetwork : nn.Module<Tensor, Tensor>
{
    private readonly List<LibQuidCell> _cells = new();
    public float DeltaTime { get; set; } = 0.1f;

    public LibQuidNetwork(List<ConnectionMap> maps) : base(nameof(LibQuidNetwork))
    {
        for (int i = 0; i < maps.Count; i++)
        {
            var cell = new LibQuidCell(maps[i]);
            _cells.Add(cell);
            register_module($"cell_{i}", cell);
        }
    }

    public override Tensor forward(Tensor input)
    {
        var batchSize = input.shape[0];
        var timeSteps = input.shape[1];

        var h_states = _cells.Select(c =>
            zeros(new long[] { batchSize, c.TargetSize }, dtype: ScalarType.Float32, device: input.device)
        ).ToList();

        for (int t = 0; t < timeSteps; t++)
        {
            using var x_t = input.select(1, t);
            Tensor currentLayerInput = x_t;

            for (int i = 0; i < _cells.Count; i++)
            {
                using var dh = _cells[i].forward(h_states[i], currentLayerInput);
                var h_next = h_states[i].add(dh.mul(DeltaTime));

                h_states[i].Dispose();
                h_states[i] = h_next;
                currentLayerInput = h_states[i];
            }
        }

        Tensor lastOutput = h_states.Last().alias();
        foreach (var h in h_states) h.Dispose();

        return lastOutput;
    }

    public float TrainStep(Tensor input, Tensor target, optim.Optimizer optimizer)
    {
        train();
        optimizer.zero_grad();

        using var pred = forward(input);
        using var loss = nn.functional.mse_loss(pred, target);

        loss.backward();

        // Aplicación manual obligatoria después del backward
        ApplyGradientMasks();

        optimizer.step();

        return loss.item<float>();
    }

    private void ApplyGradientMasks()
    {
        foreach (var cell in _cells)
        {
            cell.ApplyGradientMask();
        }
    }

    public string ComputeTopologyHash(List<ConnectionMap> maps)
    {
        var json = JsonSerializer.Serialize(maps);
        using var sha = SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(Encoding.UTF8.GetBytes(json)));
    }
}