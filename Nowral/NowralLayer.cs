using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using NowLib;

namespace Nowral;

public class NowralLayer : nn.Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly Parameter bias;
    private readonly IActivation _activation;
    private readonly string _maskBufferName = "topology_mask";

    public NowralLayer(ConnectionMap map, IActivation activation) : base(nameof(NowralLayer))
    {
        _activation = activation;
        weight = nn.Parameter(randn(new long[] { map.TargetSize, map.SourceSize }, dtype: ScalarType.Float32) * 0.1);
        bias = nn.Parameter(zeros(new long[] { map.TargetSize }, dtype: ScalarType.Float32));

        var mask = map.GenerateMask();
        register_buffer(_maskBufferName, mask);
        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }

    private Tensor Mask => get_buffer(_maskBufferName);

    public override Tensor forward(Tensor input)
    {
        // El 'using' solo para el resultado de la multiplicación, no para Mask ni weight
        using var maskedWeight = weight.mul(Mask);
        using var z = nn.functional.linear(input, maskedWeight, bias);
        return _activation.Activate(z);
    }

    public void ApplyGradientMask()
    {
        var grad = weight.grad;
        if (grad is not null) grad.mul_(Mask); // Operación in-place segura
    }
}