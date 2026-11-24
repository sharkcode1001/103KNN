using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Encoding;

/// <summary>
/// Codificador temporal Time-to-First-Spike (TTFS) / Rank-Order Coding.
/// Los valores con mayor activación disparan primero (menor timestamp), permitiendo
/// que la red procese la información crítica con prioridad en el tiempo.
/// </summary>
public sealed class TimeToFirstSpikeEncoder : ISpikeEncoder
{
    private readonly float _minActivationThreshold;

    public TimeToFirstSpikeEncoder(float minActivationThreshold = 0.05f)
    {
        _minActivationThreshold = minActivationThreshold;
    }

    public IReadOnlyList<Spike> Encode(ReadOnlySpan<float> continuousVector, float timeWindow = 1.0f)
    {
        if (continuousVector.Length == 0)
            return Array.Empty<Spike>();

        // Encontrar valor máximo para normalización
        float maxVal = 0.00001f;
        for (int i = 0; i < continuousVector.Length; i++)
        {
            float abs = MathF.Abs(continuousVector[i]);
            if (abs > maxVal) maxVal = abs;
        }

        var spikes = new List<Spike>(continuousVector.Length);

        for (int i = 0; i < continuousVector.Length; i++)
        {
            float val = continuousVector[i];
            float normalized = MathF.Abs(val) / maxVal;

            if (normalized < _minActivationThreshold)
                continue;

            // Cuanto mayor la magnitud, antes dispara: t = t_max * (1.0 - normalized)
            float timestamp = timeWindow * (1.0f - normalized);
            int polarity = val >= 0 ? 1 : -1;

            spikes.Add(new Spike(i, timestamp, normalized, polarity));
        }

        // Ordenar temporalmente (Rank-Order)
        spikes.Sort();
        return spikes;
    }
}
