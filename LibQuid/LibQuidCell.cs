using NowLib;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace LibQuid;

public class LibQuidCell : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly Parameter _w;
    private readonly Parameter _b;
    private readonly Parameter _tau;
    private readonly string _maskName = "topo_mask";

    public int SourceSize { get; private set; }
    public int TargetSize { get; private set; }

    private readonly IActivation _activation;

    public LibQuidCell(ConnectionMap map, IActivation? activation = null) : base(nameof(LibQuidCell))
    {
        _activation = activation ?? new TanhActivation();
        SourceSize = map.SourceSize;
        TargetSize = map.TargetSize;

        var mask = map.GenerateMask();
        register_buffer(_maskName, mask);

        // Inicialización Kaiming
        var rawWeight = randn(new long[] { TargetSize, SourceSize }) * (float)Math.Sqrt(2.0 / SourceSize);

        using (no_grad())
        {
            rawWeight.mul_(mask);
        }

        _w = nn.Parameter(rawWeight);
        _b = nn.Parameter(zeros(new long[] { TargetSize }));
        _tau = nn.Parameter(ones(new long[] { TargetSize }));

        register_parameter("weight", _w);
        register_parameter("bias", _b);
        register_parameter("tau", _tau);
    }

    private Tensor Mask => get_buffer(_maskName);

    public override Tensor forward(Tensor h, Tensor x)
    {
        using var v = nn.functional.linear(x, _w, _b);
        using var activation = _activation.Activate(v); // Uso de la interfaz unificada

        using var safeTau = _tau.clamp(min: 1e-4);
        using var invTau = safeTau.reciprocal();

        return invTau.neg().mul(h.sub(activation));
    }

    public void ApplyGradientMask()
    {
        using var scope = no_grad();
        var grad = _w.grad;
        if (grad is not null)
        {
            grad.mul_(Mask);
        }
    }
}