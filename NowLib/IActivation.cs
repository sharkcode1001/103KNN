using TorchSharp;
using static TorchSharp.torch;

namespace NowLib;

public interface IActivation
{
    Tensor Activate(Tensor x);
}

public class ReLUActivation : IActivation { public Tensor Activate(Tensor x) => nn.functional.relu(x); }
public class LeakyReLUActivation : IActivation { public Tensor Activate(Tensor x) => nn.functional.leaky_relu(x, 0.01); }
public class SigmoidActivation : IActivation { public Tensor Activate(Tensor x) => torch.sigmoid(x); }
public class TanhActivation : IActivation { public Tensor Activate(Tensor x) => torch.tanh(x); }