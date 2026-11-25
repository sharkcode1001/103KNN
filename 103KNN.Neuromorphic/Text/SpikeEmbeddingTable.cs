using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Encoding;

namespace _103KNN.Neuromorphic.Text;

/// <summary>
/// Tabla de embeddings que mapea identificadores de tokens a representaciones latentes y trenes de espigas temporales.
/// Soporta expansión dinámica de vocabulario para admitir nuevos tokens durante el entrenamiento.
/// </summary>
public sealed class SpikeEmbeddingTable
{
    private readonly int _embeddingDim;
    private readonly List<float[]> _embeddings = new();
    private readonly ISpikeEncoder _spikeEncoder;
    private readonly Random _rand;
    private readonly float _scale;

    public SpikeEmbeddingTable(int initialVocabSize, int embeddingDim, Random? rng = null)
    {
        _embeddingDim = embeddingDim;
        _rand = rng ?? new Random(2026);
        _scale = 1.0f / MathF.Sqrt(embeddingDim);
        _spikeEncoder = new TimeToFirstSpikeEncoder();

        EnsureCapacity(initialVocabSize);
    }

    public int VocabSize => _embeddings.Count;
    public int EmbeddingDim => _embeddingDim;

    public void EnsureCapacity(int size)
    {
        while (_embeddings.Count < size)
        {
            float[] vec = new float[_embeddingDim];
            for (int d = 0; d < _embeddingDim; d++)
            {
                vec[d] = (float)((_rand.NextDouble() * 2.0 - 1.0) * _scale);
            }
            _embeddings.Add(vec);
        }
    }

    public float[] GetVector(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _embeddings.Count)
            return _embeddings[SpikeTokenizer.UnkId];

        return _embeddings[tokenId];
    }

    public void SetVector(int tokenId, float[] vector)
    {
        EnsureCapacity(tokenId + 1);
        if (tokenId >= 0 && vector.Length == _embeddingDim)
        {
            Array.Copy(vector, _embeddings[tokenId], _embeddingDim);
        }
    }

    /// <summary>
    /// Codifica directamente un token ID a un tren de espigas temporales (Rank-Order / TTFS).
    /// </summary>
    public IReadOnlyList<Spike> GetSpikes(int tokenId, float timeWindow = 1.0f)
    {
        var vec = GetVector(tokenId);
        return _spikeEncoder.Encode(vec, timeWindow);
    }
}
