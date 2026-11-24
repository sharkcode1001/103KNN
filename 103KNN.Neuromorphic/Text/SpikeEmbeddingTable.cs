using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Encoding;

namespace _103KNN.Neuromorphic.Text;

/// <summary>
/// Tabla de embeddings que mapea identificadores de tokens a representaciones latentes y trenes de espigas temporales.
/// </summary>
public sealed class SpikeEmbeddingTable
{
    private readonly int _vocabSize;
    private readonly int _embeddingDim;
    private readonly float[][] _embeddings;
    private readonly ISpikeEncoder _spikeEncoder;

    public SpikeEmbeddingTable(int vocabSize, int embeddingDim, Random? rng = null)
    {
        _vocabSize = vocabSize;
        _embeddingDim = embeddingDim;
        _embeddings = new float[vocabSize][];
        _spikeEncoder = new TimeToFirstSpikeEncoder();

        var rand = rng ?? new Random(2026);
        float scale = 1.0f / MathF.Sqrt(embeddingDim);

        for (int i = 0; i < vocabSize; i++)
        {
            _embeddings[i] = new float[embeddingDim];
            for (int d = 0; d < embeddingDim; d++)
            {
                _embeddings[i][d] = (float)((rand.NextDouble() * 2.0 - 1.0) * scale);
            }
        }
    }

    public int VocabSize => _vocabSize;
    public int EmbeddingDim => _embeddingDim;

    public float[] GetVector(int tokenId)
    {
        if (tokenId < 0 || tokenId >= _vocabSize)
            return _embeddings[SpikeTokenizer.UnkId];

        return _embeddings[tokenId];
    }

    public void SetVector(int tokenId, float[] vector)
    {
        if (tokenId >= 0 && tokenId < _vocabSize && vector.Length == _embeddingDim)
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
