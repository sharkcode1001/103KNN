using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Layers;

namespace _103KNN.Neuromorphic.Plasticity;

/// <summary>
/// Entrenador sináptico online no supervisado / semi-supervisado basado en la regla de plasticidad STDP.
/// Permite que la red aprenda correlaciones causales temporales sin gradientes de retropropagación.
/// </summary>
public sealed class StdpSynapticTrainer
{
    private readonly StdpConfig _config;

    public StdpSynapticTrainer(StdpConfig? config = null)
    {
        _config = config ?? new StdpConfig();
    }

    public StdpConfig Config => _config;

    /// <summary>
    /// Aplica la regla STDP sobre una capa sináptica a partir de las espigas pre-sinápticas y el disparo post-sináptico.
    /// Retorna el cambio neto acumulado en las sinapsis (deltaW absoluto).
    /// </summary>
    public float ApplyStdp(SpikeSynapticLayer layer, IReadOnlyList<Spike> preSpikes)
    {
        float totalWeightDelta = 0.0f;

        for (int postIdx = 0; postIdx < layer.OutputDim; postIdx++)
        {
            var postNeuron = layer.Neurons[postIdx];
            if (!postNeuron.HasFiredInEpisode)
                continue;

            float tPost = postNeuron.FirstSpikeTime;

            foreach (var pre in preSpikes)
            {
                if (pre.FeatureIndex < 0 || pre.FeatureIndex >= layer.InputDim)
                    continue;

                float tPre = pre.Timestamp;
                float deltaT = tPost - tPre;

                float deltaW = 0.0f;

                if (deltaT > 0.0f)
                {
                    // Pre disparó antes de Post -> Potenciación Causal (LTP)
                    deltaW = _config.APlus * MathF.Exp(-deltaT / _config.TauPlus);
                }
                else if (deltaT < 0.0f)
                {
                    // Post disparó antes de Pre -> Depresión Anti-Causal (LTD)
                    deltaW = -_config.AMinus * MathF.Exp(deltaT / _config.TauMinus);
                }

                deltaW *= _config.LearningRate * pre.Magnitude;

                if (MathF.Abs(deltaW) > 1e-6f)
                {
                    float currentW = layer.GetWeight(postIdx, pre.FeatureIndex);
                    float newW = Math.Clamp(currentW + deltaW, _config.MinWeight, _config.MaxWeight);
                    layer.SetWeight(postIdx, pre.FeatureIndex, newW);
                    totalWeightDelta += MathF.Abs(deltaW);
                }
            }
        }

        return totalWeightDelta;
    }

    /// <summary>
    /// Aplica refuerzo supervisado o guiado por recompensa (Reward-modulated STDP / R-STDP)
    /// Potencia selectivamente las conexiones de la neurona objetivo y deprime las competidoras.
    /// </summary>
    public void ApplyReinforcedStdp(SpikeSynapticLayer layer, IReadOnlyList<Spike> preSpikes, int targetNeuronIndex, float reward = 1.0f)
    {
        for (int postIdx = 0; postIdx < layer.OutputDim; postIdx++)
        {
            float modulation = (postIdx == targetNeuronIndex) ? reward : -reward * 0.35f;

            foreach (var pre in preSpikes)
            {
                if (pre.FeatureIndex < 0 || pre.FeatureIndex >= layer.InputDim)
                    continue;

                float currentW = layer.GetWeight(postIdx, pre.FeatureIndex);
                float dw = _config.APlus * modulation * pre.Magnitude * 0.05f;
                float newW = Math.Clamp(currentW + dw, _config.MinWeight, _config.MaxWeight);
                layer.SetWeight(postIdx, pre.FeatureIndex, newW);
            }
        }
    }
}
