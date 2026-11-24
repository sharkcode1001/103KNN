using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Encoding;

/// <summary>
/// Contrato para transformar vectores de características continuas o embeddings a trenes de espigas temporales.
/// </summary>
public interface ISpikeEncoder
{
    /// <summary>
    /// Codifica un vector denso (embedding de token o vector de estado) en una secuencia de espigas ordenadas en el tiempo.
    /// </summary>
    IReadOnlyList<Spike> Encode(ReadOnlySpan<float> continuousVector, float timeWindow = 1.0f);
}
