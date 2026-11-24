using System.Diagnostics;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Layers;

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
/// Genera texto token a token mediante dinámica de espigas temporales sin necesidad de KV-cache.
/// </summary>
public sealed class SpikeTextGenerator
{
    private readonly SpikeTokenizer _tokenizer;
    private readonly SpikeEmbeddingTable _embeddings;
    private readonly CausalSpikeRecurrentLayer _recurrentLayer;
    private readonly SpikeSynapticLayer _vocabHead;
    private readonly int _embeddingDim;
    private readonly int _hiddenDim;

    public SpikeTextGenerator(int embeddingDim = 32, int hiddenDim = 64, IEnumerable<string>? initialCorpus = null)
    {
        _embeddingDim = embeddingDim;
        _hiddenDim = hiddenDim;
        _tokenizer = new SpikeTokenizer();

        var corpus = (initialCorpus ?? GetDefaultReasoningCorpus()).ToList();
        _tokenizer.BuildVocabulary(corpus);

        int vocabSize = _tokenizer.VocabSize;
        _embeddings = new SpikeEmbeddingTable(vocabSize, embeddingDim);

        var config = new CognitiveConfig
        {
            BaseThreshold = 0.30f,
            MembraneDecay = 0.88f,
            LateralInhibitionFactor = 0.15f
        };

        _recurrentLayer = new CausalSpikeRecurrentLayer(embeddingDim, hiddenDim, config);
        _vocabHead = new SpikeSynapticLayer(hiddenDim, vocabSize, config, gain: 2.0f);

        // Pre-entrenar con el corpus inicial de razonamiento
        TrainOnCorpus(corpus, epochs: 12);
    }

    public SpikeTokenizer Tokenizer => _tokenizer;
    public int VocabSize => _tokenizer.VocabSize;
    public int EmbeddingDim => _embeddingDim;
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Entrena las asociaciones sinápticas causales sobre secuencias de texto utilizando aprendizaje asociativo Hebbiano / STDP.
    /// </summary>
    public void TrainOnCorpus(IEnumerable<string> sentences, int epochs = 10, float learningRate = 0.15f)
    {
        var list = sentences.ToList();
        _tokenizer.BuildVocabulary(list);

        for (int ep = 0; ep < epochs; ep++)
        {
            foreach (var sentence in list)
            {
                var tokens = _tokenizer.Encode(sentence, addBos: true, addEos: true);
                if (tokens.Count < 2) continue;

                _recurrentLayer.ResetState();

                for (int t = 0; t < tokens.Count - 1; t++)
                {
                    int currentToken = tokens[t];
                    int targetNextToken = tokens[t + 1];

                    var inSpikes = _embeddings.GetSpikes(currentToken);
                    var hiddenSpikes = _recurrentLayer.Step(inSpikes);

                    // Refuerzo Hebbiano causal hacia la neurona del token objetivo
                    foreach (var hSpike in hiddenSpikes)
                    {
                        if (hSpike.FeatureIndex >= 0 && hSpike.FeatureIndex < _hiddenDim)
                        {
                            // Potenciar conexión hacia la palabra objetivo
                            float currentW = _vocabHead.GetWeight(targetNextToken, hSpike.FeatureIndex);
                            float newW = Math.Clamp(currentW + (learningRate * hSpike.Magnitude), -2.0f, 2.5f);
                            _vocabHead.SetWeight(targetNextToken, hSpike.FeatureIndex, newW);
                        }
                    }

                    // Depresión sináptica de regularización para tokens que no debían disparar
                    for (int v = 0; v < _tokenizer.VocabSize; v++)
                    {
                        if (v != targetNextToken && _vocabHead.Neurons[v].HasFiredInEpisode)
                        {
                            foreach (var hSpike in hiddenSpikes)
                            {
                                float w = _vocabHead.GetWeight(v, hSpike.FeatureIndex);
                                _vocabHead.SetWeight(v, hSpike.FeatureIndex, Math.Clamp(w - (learningRate * 0.25f), -2.0f, 2.5f));
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Genera la continuación de un texto de manera autorregresiva token a token.
    /// </summary>
    public SpikeGenerationResult Generate(string prompt, int maxNewTokens = 15, float temperature = 0.7f)
    {
        var sw = Stopwatch.StartNew();
        _recurrentLayer.ResetState();

        var promptTokens = _tokenizer.Encode(prompt, addBos: true, addEos: false);
        int totalSpikes = 0;

        // 1. Ingesta del prompt para calentar la memoria de trabajo de las neuronas recurrentes
        foreach (int token in promptTokens)
        {
            var inSpikes = _embeddings.GetSpikes(token);
            totalSpikes += inSpikes.Count;
            var hiddenSpikes = _recurrentLayer.Step(inSpikes);
            totalSpikes += hiddenSpikes.Count;
        }

        var generatedTokens = new List<int>();
        int currentTokenId = promptTokens.Count > 0 ? promptTokens[^1] : SpikeTokenizer.BosId;
        var rand = new Random();

        // 2. Bucle Autorregresivo de Generación
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

            // Selección de siguiente token (TTFS / Potencial de Membrana)
            int nextToken = SelectNextToken(temperature, rand);

            // Detener si emite <eos> o repite de forma nula
            if (nextToken is SpikeTokenizer.EosId or SpikeTokenizer.PadId)
                break;

            generatedTokens.Add(nextToken);
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

    private int SelectNextToken(float temperature, Random rand)
    {
        // Enfoque TTFS: La primera neurona de vocabulario que disparó
        float bestTime = float.PositiveInfinity;
        int bestNeuron = -1;

        for (int v = 0; v < _tokenizer.VocabSize; v++)
        {
            if (v is SpikeTokenizer.PadId or SpikeTokenizer.UnkId or SpikeTokenizer.BosId)
                continue;

            float t = _vocabHead.Neurons[v].FirstSpikeTime;
            if (t < bestTime)
            {
                bestTime = t;
                bestNeuron = v;
            }
        }

        // Si alguna neurona disparó con clara prioridad temporal
        if (bestNeuron >= 0 && rand.NextDouble() > (temperature * 0.4))
        {
            return bestNeuron;
        }

        // Muestreo ponderado por potencial de membrana
        float[] potentials = new float[_tokenizer.VocabSize];
        float sum = 0.0001f;

        for (int v = 0; v < _tokenizer.VocabSize; v++)
        {
            if (v is SpikeTokenizer.PadId or SpikeTokenizer.UnkId or SpikeTokenizer.BosId)
                continue;

            float val = MathF.Max(0.01f, _vocabHead.Neurons[v].Voltage);
            potentials[v] = MathF.Pow(val, 1.0f / MathF.Max(0.1f, temperature));
            sum += potentials[v];
        }

        float r = (float)(rand.NextDouble() * sum);
        float accum = 0.0f;
        for (int v = 0; v < _tokenizer.VocabSize; v++)
        {
            accum += potentials[v];
            if (accum >= r) return v;
        }

        return bestNeuron >= 0 ? bestNeuron : SpikeTokenizer.EosId;
    }

    public static List<string> GetDefaultReasoningCorpus()
    {
        return new List<string>
        {
            "el sol produce energia limpia y radiacion solar para los paneles",
            "la radiacion solar genera electricidad renovable de alta eficiencia",
            "si hay nubes densas entonces la radiacion disminuye en la superficie",
            "si la radiacion disminuye entonces la produccion electrica desciende",
            "el viento desplaza las nubes y altera el patron de sombra",
            "la red neuromorfica razona mediante espigas temporales eficientes",
            "las neuronas lif integran potencial y disparan por orden de rango",
            "la inhibicion lateral selecciona la mejor decision y suprime el ruido",
            "el orden de llegada de las espigas codifica el significado conceptual",
            "la energia solar es el futuro de la generacion energetica sostenible",
            "si la produccion solar aumenta entonces la red electrica se estabiliza"
        };
    }
}
