using System.Diagnostics;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Layers;
using _103KNN.Neuromorphic.Training;

namespace _103KNN.Neuromorphic.Text;

public sealed record SpikeGenerationResult(
    string Prompt,
    string GeneratedText,
    IReadOnlyList<int> GeneratedTokens,
    int TotalSpikesFired,
    int TokensCount,
    TimeSpan ElapsedTime,
    double MicrosecondsPerToken
);

/// <summary>
/// Generador de texto y razonamiento autorregresivo neuromórfico (Spike-LLM).
/// Genera texto token a token mediante dinámica de espigas temporales, sin KV-cache,
/// con inhibición de recencia (repetition penalty bio-inspirado), transiciones causales de orden superior (bigramas y trigramas)
/// y memoria asociativa de contexto semántico.
/// </summary>
public sealed class SpikeTextGenerator
{
    private readonly SpikeTokenizer _tokenizer;
    private readonly SpikeEmbeddingTable _embeddings;
    private readonly CausalSpikeRecurrentLayer _recurrentLayer;
    private readonly SpikeSynapticLayer _vocabHead;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;

    // Transición sintáctica causal: [prevToken -> {nextToken, weight}]
    private readonly Dictionary<int, Dictionary<int, float>> _causalBigrams = new();

    // Transición de orden superior: [(prev2, prev1) -> {nextToken, weight}]
    private readonly Dictionary<long, Dictionary<int, float>> _causalTrigrams = new();

    // Afinidad contextual: [promptKeywordToken -> {relatedVocabToken, weight}]
    private readonly Dictionary<int, Dictionary<int, float>> _contextAffinities = new();

    // Índice asociativo de conocimiento para alineación de intención y respuesta (Memoria Asociativa SNN)
    private readonly List<(HashSet<int> QueryTokens, List<int> AnswerTokens)> _qaKnowledgeBank = new();

    public string ActiveModelName { get; set; } = "103KNN BioNet (Nativo)";

    public SpikeTextGenerator(
        int embeddingDim = 32,
        int hiddenDim = 64,
        IEnumerable<string>? trainingCorpus = null,
        string modelName = "103KNN BioNet (Nativo)")
    {
        ActiveModelName = modelName;
        _embeddingDim = embeddingDim;
        _hiddenDim = hiddenDim;
        _tokenizer = new SpikeTokenizer();

        var corpus = (trainingCorpus ?? NeuromorphicKnowledgeTrainer.GetComprehensiveNativeCorpus()).ToList();
        _tokenizer.BuildVocabulary(corpus);

        int vocabSize = Math.Max(256, _tokenizer.VocabSize + 64);
        _embeddings = new SpikeEmbeddingTable(vocabSize, embeddingDim);

        var config = new CognitiveConfig
        {
            BaseThreshold = 0.25f,
            MembraneDecay = 0.88f,
            LateralInhibitionFactor = 0.15f
        };

        _recurrentLayer = new CausalSpikeRecurrentLayer(embeddingDim, hiddenDim, config);
        _vocabHead = new SpikeSynapticLayer(hiddenDim, vocabSize, config, gain: 2.2f);

        // Pre-entrenar con el corpus estructurado nativo
        TrainOnCorpusDetailed(corpus, epochs: 14, learningRate: 0.25f);
    }

    public SpikeTokenizer Tokenizer => _tokenizer;
    public int VocabSize => _tokenizer.VocabSize;
    public int EmbeddingDim => _embeddingDim;
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Entrena las asociaciones sinápticas causales, bigramas, trigramas y afinidad conceptual sobre el corpus.
    /// </summary>
    public int TrainOnCorpusDetailed(
        IEnumerable<string> sentences,
        int epochs = 14,
        float learningRate = 0.25f,
        Action<int, int>? epochCallback = null)
    {
        var list = sentences.ToList();
        _tokenizer.BuildVocabulary(list);

        int currentVocab = _tokenizer.VocabSize;
        if (currentVocab > _vocabHead.OutputDim)
        {
            int newSize = currentVocab + 32;
            _embeddings.EnsureCapacity(newSize);
            _vocabHead.ExpandOutputDim(newSize);
        }

        int synapsesUpdated = 0;

        for (int ep = 0; ep < epochs; ep++)
        {
            foreach (var sentence in list)
            {
                var tokens = _tokenizer.Encode(sentence, addBos: true, addEos: true);
                if (tokens.Count < 2) continue;

                // Indexar pares Q&A para memoria asociativa si contiene "asistente:"
                if (ep == 0 && sentence.Contains("asistente:", StringComparison.OrdinalIgnoreCase))
                {
                    IndexQaPair(sentence);
                }

                _recurrentLayer.ResetState();

                for (int t = 0; t < tokens.Count - 1; t++)
                {
                    int currentToken = tokens[t];
                    int targetNextToken = tokens[t + 1];

                    // 1. Reforzar Bigrama causal
                    if (!_causalBigrams.TryGetValue(currentToken, out var nextMap))
                    {
                        nextMap = new Dictionary<int, float>();
                        _causalBigrams[currentToken] = nextMap;
                    }
                    nextMap[targetNextToken] = nextMap.GetValueOrDefault(targetNextToken, 0.0f) + 0.35f;

                    // 2. Reforzar Trigrama causal si t >= 1
                    if (t >= 1)
                    {
                        int prevToken = tokens[t - 1];
                        long triKey = ((long)prevToken << 32) | (uint)currentToken;
                        if (!_causalTrigrams.TryGetValue(triKey, out var triMap))
                        {
                            triMap = new Dictionary<int, float>();
                            _causalTrigrams[triKey] = triMap;
                        }
                        triMap[targetNextToken] = triMap.GetValueOrDefault(targetNextToken, 0.0f) + 0.50f;
                    }

                    // 3. Propagación de espigas y refuerzo Hebbiano en la capa de salida
                    var inSpikes = _embeddings.GetSpikes(currentToken);
                    var hiddenSpikes = _recurrentLayer.Step(inSpikes);

                    foreach (var hSpike in hiddenSpikes)
                    {
                        if (hSpike.FeatureIndex >= 0 && hSpike.FeatureIndex < _hiddenDim)
                        {
                            float currentW = _vocabHead.GetWeight(targetNextToken, hSpike.FeatureIndex);
                            float newW = Math.Clamp(currentW + (learningRate * hSpike.Magnitude), -2.0f, 3.5f);
                            _vocabHead.SetWeight(targetNextToken, hSpike.FeatureIndex, newW);
                            synapsesUpdated++;
                        }
                    }
                }
            }

            epochCallback?.Invoke(ep + 1, synapsesUpdated);
        }

        return synapsesUpdated;
    }

    private void IndexQaPair(string sentence)
    {
        int asisIdx = sentence.IndexOf("asistente:", StringComparison.OrdinalIgnoreCase);
        if (asisIdx <= 0) return;

        string qPart = sentence.Substring(0, asisIdx).Replace("usuario:", "").Trim();
        string aPart = sentence.Substring(asisIdx + "asistente:".Length).Trim();

        var qTokens = _tokenizer.Encode(qPart, false, false).Where(id => id > 3).ToHashSet();
        var aTokens = _tokenizer.Encode(aPart, false, false);

        if (qTokens.Count > 0 && aTokens.Count > 0)
        {
            _qaKnowledgeBank.Add((qTokens, aTokens));

            // Fortalecer transición hacia EOS al final de la respuesta aprendida
            int lastToken = aTokens[^1];
            if (!_causalBigrams.TryGetValue(lastToken, out var nextMap))
            {
                nextMap = new Dictionary<int, float>();
                _causalBigrams[lastToken] = nextMap;
            }
            nextMap[SpikeTokenizer.EosId] = nextMap.GetValueOrDefault(SpikeTokenizer.EosId, 0.0f) + 3.5f;

            // Fortalecer afinidades conceptuales entre palabras clave de la pregunta y la respuesta
            foreach (int qTok in qTokens)
            {
                if (!_contextAffinities.TryGetValue(qTok, out var affMap))
                {
                    affMap = new Dictionary<int, float>();
                    _contextAffinities[qTok] = affMap;
                }

                foreach (int aTok in aTokens)
                {
                    if (aTok > 3)
                    {
                        affMap[aTok] = affMap.GetValueOrDefault(aTok, 0.0f) + 0.45f;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Modo de diálogo conversacional directo (Chat SNN).
    /// </summary>
    public SpikeGenerationResult Chat(string userMessage, int maxNewTokens = 35, float temperature = 0.45f)
    {
        string prompt = $"usuario: {userMessage} asistente:";
        return Generate(prompt, maxNewTokens, temperature, minNewTokens: 5);
    }

    /// <summary>
    /// Genera la continuación de un texto de manera autorregresiva token a token.
    /// </summary>
    public SpikeGenerationResult Generate(string prompt, int maxNewTokens = 25, float temperature = 0.45f, int minNewTokens = 3)
    {
        var sw = Stopwatch.StartNew();
        _recurrentLayer.ResetState();

        var promptTokens = _tokenizer.Encode(prompt, addBos: true, addEos: false);
        int totalSpikes = 0;

        // Conjunto de tokens clave del prompt para condicionamiento semántico
        var promptKeywords = promptTokens.Where(t => t > 3).ToHashSet();

        // 1. Ingesta del prompt para calentar la memoria de trabajo de las neuronas recurrentes
        foreach (int token in promptTokens)
        {
            var inSpikes = _embeddings.GetSpikes(token);
            totalSpikes += inSpikes.Count;
            var hiddenSpikes = _recurrentLayer.Step(inSpikes);
            totalSpikes += hiddenSpikes.Count;
        }

        // 2. Consulta a Memoria Asociativa SNN (Búsqueda de concepto más cercano en el banco de conocimiento)
        List<int>? associativeHint = FindBestAssociativeMatch(promptKeywords);

        var generatedTokens = new List<int>();
        var recentTokensQueue = new Queue<int>(capacity: 8);
        int prevPrevToken = promptTokens.Count > 1 ? promptTokens[^2] : SpikeTokenizer.BosId;
        int currentTokenId = promptTokens.Count > 0 ? promptTokens[^1] : SpikeTokenizer.BosId;
        var rand = new Random();

        // 3. Bucle Autorregresivo de Generación
        for (int i = 0; i < maxNewTokens; i++)
        {
            var inSpikes = _embeddings.GetSpikes(currentTokenId);
            totalSpikes += inSpikes.Count;

            var hiddenSpikes = _recurrentLayer.Step(inSpikes);
            totalSpikes += hiddenSpikes.Count;

            // Proyección al vocabulario por acumulación de espigas
            _vocabHead.Reset();
            var vocabSpikes = _vocabHead.Forward(hiddenSpikes);
            totalSpikes += vocabSpikes.Count;

            // Selección de siguiente token modulada por dinámica biofísica, n-gramas causales y contexto
            int nextToken = SelectNextToken(
                prevPrevToken,
                currentTokenId,
                promptKeywords,
                associativeHint,
                i,
                recentTokensQueue,
                temperature,
                rand
            );

            // Detener si emite fin de texto o token nulo tras cumplir el mínimo
            if (nextToken is SpikeTokenizer.EosId or SpikeTokenizer.PadId)
            {
                if (i >= minNewTokens)
                    break;

                nextToken = SelectFallbackToken(recentTokensQueue, rand);
            }

            string tokStr = _tokenizer.GetToken(nextToken);

            // Detener si emite un punto o signo de cierre (.) o delimitador de turno
            if (i >= minNewTokens && (tokStr is "." or "!" or "?" or "usuario:" or "usuario" or "asistente:" or "<eos>"))
            {
                if (tokStr is "." or "!" or "?")
                    generatedTokens.Add(nextToken);
                break;
            }

            // Si se completó la respuesta guiada por la memoria asociativa de conocimiento
            if (associativeHint != null && i >= associativeHint.Count && i >= minNewTokens)
            {
                break;
            }

            generatedTokens.Add(nextToken);
            recentTokensQueue.Enqueue(nextToken);
            if (recentTokensQueue.Count > 6)
                recentTokensQueue.Dequeue();

            prevPrevToken = currentTokenId;
            currentTokenId = nextToken;
        }

        sw.Stop();

        string fullGenerated = _tokenizer.Decode(generatedTokens);
        double usPerToken = generatedTokens.Count > 0 ? sw.Elapsed.TotalMicroseconds / generatedTokens.Count : 0.0;

        return new SpikeGenerationResult(
            Prompt: prompt,
            GeneratedText: fullGenerated,
            GeneratedTokens: generatedTokens,
            TotalSpikesFired: totalSpikes,
            TokensCount: generatedTokens.Count,
            ElapsedTime: sw.Elapsed,
            MicrosecondsPerToken: usPerToken
        );
    }

    private List<int>? FindBestAssociativeMatch(HashSet<int> promptKeywords)
    {
        if (promptKeywords.Count == 0 || _qaKnowledgeBank.Count == 0) return null;

        List<int>? bestAnswer = null;
        float bestScore = 0.0f;

        foreach (var (qSet, aList) in _qaKnowledgeBank)
        {
            int overlap = 0;
            foreach (var tok in promptKeywords)
            {
                if (qSet.Contains(tok)) overlap++;
            }

            if (overlap > 0)
            {
                float score = (float)overlap / MathF.Sqrt(promptKeywords.Count * qSet.Count);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestAnswer = aList;
                }
            }
        }

        return bestScore >= 0.25f ? bestAnswer : null;
    }

    private int SelectNextToken(
        int prevPrevToken,
        int previousToken,
        HashSet<int> promptKeywords,
        List<int>? associativeHint,
        int stepIndex,
        Queue<int> recentTokens,
        float temperature,
        Random rand)
    {
        int vocabSize = _vocabHead.OutputDim;
        float sum = 0.0001f;

        var candidateIndices = new HashSet<int>(_vocabHead.ActiveExcitedNeurons);

        // Añadir candidatos de bigramas
        if (_causalBigrams.TryGetValue(previousToken, out var nextMap))
        {
            foreach (int nextTok in nextMap.Keys)
            {
                if (nextTok >= 0 && nextTok < vocabSize)
                    candidateIndices.Add(nextTok);
            }
        }

        // Añadir candidatos de trigramas
        long triKey = ((long)prevPrevToken << 32) | (uint)previousToken;
        Dictionary<int, float>? triMap = null;
        if (_causalTrigrams.TryGetValue(triKey, out triMap))
        {
            foreach (int nextTok in triMap.Keys)
            {
                if (nextTok >= 0 && nextTok < vocabSize)
                    candidateIndices.Add(nextTok);
            }
        }

        // Añadir candidato de la memoria asociativa guiada si coincide la posición
        if (associativeHint != null && stepIndex < associativeHint.Count)
        {
            int hintTok = associativeHint[stepIndex];
            if (hintTok >= 0 && hintTok < vocabSize)
                candidateIndices.Add(hintTok);
        }

        if (candidateIndices.Count == 0)
        {
            return SelectFallbackToken(recentTokens, rand);
        }

        var scores = new Dictionary<int, float>(candidateIndices.Count);

        foreach (int v in candidateIndices)
        {
            if (v is SpikeTokenizer.PadId or SpikeTokenizer.UnkId or SpikeTokenizer.BosId)
                continue;

            // 1. Potencial integrado de la neurona LIF
            float voltage = MathF.Max(0.0f, _vocabHead.Neurons[v].Voltage);

            // 2. Transición causal de bigrama
            float bigramAffinity = 0.0f;
            if (nextMap != null && nextMap.TryGetValue(v, out float bVal))
                bigramAffinity = bVal;

            // 3. Transición causal de trigrama
            float trigramAffinity = 0.0f;
            if (triMap != null && triMap.TryGetValue(v, out float tVal))
                trigramAffinity = tVal;

            // 4. Afinidad semántica con palabras del prompt
            float contextAffinity = 0.0f;
            foreach (int pKey in promptKeywords)
            {
                if (_contextAffinities.TryGetValue(pKey, out var affMap) && affMap.TryGetValue(v, out float aVal))
                {
                    contextAffinity += aVal;
                }
            }

            // 5. Coincidencia con la memoria asociativa de conocimiento
            float hintAffinity = 0.0f;
            if (associativeHint != null)
            {
                if (stepIndex < associativeHint.Count && associativeHint[stepIndex] == v)
                    hintAffinity = (stepIndex == 0) ? 4.5f : 3.0f;
                else if (associativeHint.Contains(v))
                    hintAffinity = 0.5f;
            }

            // 6. Inhibición biológica de recencia (repetition penalty)
            float recencyPenalty = recentTokens.Contains(v) ? 0.12f : 1.0f;

            float combined = ((voltage * 0.30f) + (bigramAffinity * 0.30f) + (trigramAffinity * 0.40f) + (contextAffinity * 0.35f) + (hintAffinity * 0.45f)) * recencyPenalty;
            if (combined > 0.001f)
            {
                float score = MathF.Pow(combined, 1.0f / MathF.Max(0.1f, temperature));
                scores[v] = score;
                sum += score;
            }
        }

        if (scores.Count == 0)
        {
            return SelectFallbackToken(recentTokens, rand);
        }

        // Selección por consenso dominante
        float maxScore = -1.0f;
        int bestV = -1;
        foreach (var (v, sc) in scores)
        {
            if (sc > maxScore)
            {
                maxScore = sc;
                bestV = v;
            }
        }

        if (bestV >= 0 && (maxScore / sum) > 0.35f && rand.NextDouble() > (temperature * 0.4f))
        {
            return bestV;
        }

        // Muestreo estocástico proporcional
        float r = (float)(rand.NextDouble() * sum);
        float accum = 0.0f;
        foreach (var (v, sc) in scores)
        {
            accum += sc;
            if (accum >= r) return v;
        }

        return bestV >= 0 ? bestV : SelectFallbackToken(recentTokens, rand);
    }

    private int SelectFallbackToken(Queue<int> recentTokens, Random rand)
    {
        int vocabSize = _vocabHead.OutputDim;
        if (vocabSize <= 4) return SpikeTokenizer.EosId;

        for (int attempts = 0; attempts < 30; attempts++)
        {
            int candidate = rand.Next(4, Math.Min(vocabSize, _tokenizer.VocabSize));
            if (!recentTokens.Contains(candidate))
                return candidate;
        }
        return 4;
    }
}
