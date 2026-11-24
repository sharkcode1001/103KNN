using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Layers;

/// <summary>
/// Capa de Atención por Orden de Rango (Rank-Order Attention).
/// En lugar de calcular softmax(Q * K^T / sqrt(d)) denso, calcula la afinidad de atención
/// midiendo la concordancia en el orden temporal relativo de llegada de las espigas.
/// </summary>
public sealed class RankOrderAttentionLayer
{
    private readonly int _featureDim;
    private readonly int _heads;
    private readonly float[][] _prototypeRanks;

    public RankOrderAttentionLayer(int featureDim, int heads = 4, Random? rng = null)
    {
        _featureDim = featureDim;
        _heads = heads;
        _prototypeRanks = new float[heads][];

        var rand = rng ?? new Random(103);
        for (int h = 0; h < heads; h++)
        {
            _prototypeRanks[h] = new float[featureDim];
            // Asigna prioridades de rango ordinal prototípicas
            for (int i = 0; i < featureDim; i++)
            {
                _prototypeRanks[h][i] = (float)rand.NextDouble();
            }
        }
    }

    /// <summary>
    /// Modula y re-pondera un tren de espigas basándose en la coincidencia ordinal con los patrones de atención.
    /// Las espigas que llegan en el orden armónico de mayor afinidad reciben amplificación temporal.
    /// </summary>
    public IReadOnlyList<Spike> Attend(IReadOnlyList<Spike> inputSpikes)
    {
        if (inputSpikes.Count == 0)
            return inputSpikes;

        // Calcular rangos relativos normalizados [0..1] según orden de llegada
        int count = inputSpikes.Count;
        var modulatedSpikes = new List<Spike>(count);

        // Evaluar compatibilidad con las cabezas de atención ordinal
        float bestAffinity = 0.0f;
        int dominantHead = 0;

        for (int h = 0; h < _heads; h++)
        {
            float score = 0.0f;
            for (int r = 0; r < count; r++)
            {
                int feat = inputSpikes[r].FeatureIndex;
                if (feat >= 0 && feat < _featureDim)
                {
                    // Penalización si el tiempo de llegada difiere de la expectativa del prototipo
                    float expected = _prototypeRanks[h][feat];
                    float actualRank = (float)r / count;
                    score += 1.0f - MathF.Abs(expected - actualRank);
                }
            }

            if (score > bestAffinity)
            {
                bestAffinity = score;
                dominantHead = h;
            }
        }

        float attentionScale = count > 0 ? (bestAffinity / count) : 1.0f;

        // Modular espigas: aquellas que refuerzan la atención dominante disparan antes
        for (int r = 0; r < count; r++)
        {
            var original = inputSpikes[r];
            float expected = (original.FeatureIndex >= 0 && original.FeatureIndex < _featureDim)
                ? _prototypeRanks[dominantHead][original.FeatureIndex]
                : 0.5f;

            float boost = 1.0f + (attentionScale * (1.0f - expected) * 0.5f);
            float newTimestamp = original.Timestamp / boost;

            modulatedSpikes.Add(new Spike(
                original.FeatureIndex,
                newTimestamp,
                original.Magnitude * boost,
                original.Polarity
            ));
        }

        modulatedSpikes.Sort();
        return modulatedSpikes;
    }
}
