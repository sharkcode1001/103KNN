using System.Text.RegularExpressions;

namespace _103KNN.Neuromorphic.Text;

/// <summary>
/// Tokenizador para texto que mapea palabras/subpalabras a identificadores numéricos continuos.
/// Soporta construcción dinámica de vocabulario y tokens especiales de control.
/// </summary>
public sealed class SpikeTokenizer
{
    public const string PadToken = "<pad>";
    public const string UnkToken = "<unk>";
    public const string BosToken = "<bos>";
    public const string EosToken = "<eos>";

    public const int PadId = 0;
    public const int UnkId = 1;
    public const int BosId = 2;
    public const int EosId = 3;

    private readonly Dictionary<string, int> _tokenToId = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _idToToken = new();

    public SpikeTokenizer()
    {
        // Registrar tokens de control
        AddToken(PadToken);
        AddToken(UnkToken);
        AddToken(BosToken);
        AddToken(EosToken);
    }

    public int VocabSize => _idToToken.Count;

    public int AddToken(string token)
    {
        if (_tokenToId.TryGetValue(token, out int id))
            return id;

        int newId = _idToToken.Count;
        _tokenToId[token] = newId;
        _idToToken.Add(token);
        return newId;
    }

    /// <summary>
    /// Construye y expande el vocabulario a partir de un conjunto de oraciones o corpus de entrenamiento.
    /// </summary>
    public void BuildVocabulary(IEnumerable<string> corpus)
    {
        foreach (var text in corpus)
        {
            var words = TokenizeString(text);
            foreach (var w in words)
            {
                AddToken(w);
            }
        }
    }

    /// <summary>
    /// Codifica una cadena de texto en una lista de IDs de tokens.
    /// </summary>
    public List<int> Encode(string text, bool addBos = false, bool addEos = false)
    {
        var result = new List<int>();
        if (addBos) result.Add(BosId);

        var words = TokenizeString(text);
        foreach (var w in words)
        {
            if (_tokenToId.TryGetValue(w, out int id))
            {
                result.Add(id);
            }
            else
            {
                result.Add(UnkId);
            }
        }

        if (addEos) result.Add(EosId);
        return result;
    }

    /// <summary>
    /// Decodifica una secuencia de IDs de tokens de vuelta a texto legible.
    /// </summary>
    public string Decode(IReadOnlyList<int> tokenIds, bool skipSpecialTokens = true)
    {
        var words = new List<string>(tokenIds.Count);

        foreach (var id in tokenIds)
        {
            if (id < 0 || id >= _idToToken.Count)
                continue;

            if (skipSpecialTokens && id is PadId or BosId or EosId)
                continue;

            words.Add(_idToToken[id]);
        }

        return string.Join(" ", words);
    }

    public string GetToken(int id)
    {
        if (id >= 0 && id < _idToToken.Count)
            return _idToToken[id];
        return UnkToken;
    }

    private static string[] TokenizeString(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<string>();

        // Separar signos de puntuación como tokens independientes
        string formatted = Regex.Replace(text.ToLowerInvariant(), @"([.,!?;:\(\)\[\]])", " $1 ");
        return formatted.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }
}
