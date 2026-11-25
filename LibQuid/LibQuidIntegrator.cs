using TorchSharp;

namespace LibQuid;

public static class LibQuidIntegrator
{
    public static torch.Tensor Step(LibQuidCell cell, torch.Tensor h, torch.Tensor x, float dt)
    {
        // dh = derivada calculada por la celda
        using var dh = cell.forward(h, x);

        // h_next = h + dh * dt
        return h.add(dh.mul(dt));
    }
}