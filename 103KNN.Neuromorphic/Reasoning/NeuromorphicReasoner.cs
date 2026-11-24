using System.Diagnostics;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Encoding;
using _103KNN.Neuromorphic.Layers;

namespace _103KNN.Neuromorphic.Reasoning;

/// <summary>
/// Motor principal de razonamiento cognitivo neuromórfico (Brain-Inspired Cognitive Engine).
/// Implementa pensamiento dual:
/// - "Sistema 1 (Rápido)": Inferencia inmediata basada en consenso de primer orden de espiga (Time-to-First-Spike).
/// - "Sistema 2 (Deliberativo)": Bucle recurrente de integración temporal para resolver ambigüedades.
/// </summary>
public sealed class NeuromorphicReasoner
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _outputDim;
    private readonly CognitiveConfig _config;

    private readonly ISpikeEncoder _encoder;
    private readonly RankOrderAttentionLayer _attentionLayer;
    private readonly SpikeSynapticLayer _hiddenLayer;
    private readonly SpikeSynapticLayer _outputLayer;

    public NeuromorphicReasoner(
        int inputDim,
        int hiddenDim,
        int outputDim,
        CognitiveConfig? config = null,
        Random? rng = null)
    {
        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _outputDim = outputDim;
        _config = config ?? new CognitiveConfig();

        var rand = rng ?? new Random(2026);
        _encoder = new TimeToFirstSpikeEncoder();
        _attentionLayer = new RankOrderAttentionLayer(inputDim, heads: 4, rng: rand);
        _hiddenLayer = new SpikeSynapticLayer(inputDim, hiddenDim, _config, rng: rand);
        _outputLayer = new SpikeSynapticLayer(hiddenDim, outputDim, _config, rng: rand);
    }

    public int InputDim => _inputDim;
    public int HiddenDim => _hiddenDim;
    public int OutputDim => _outputDim;
    public CognitiveConfig Config => _config;
    public SpikeSynapticLayer HiddenLayer => _hiddenLayer;
    public SpikeSynapticLayer OutputLayer => _outputLayer;

    /// <summary>
    /// Ejecuta el proceso de deliberación cognitiva sobre un vector de entrada (ej. embedding o características).
    /// </summary>
    public CognitiveReasoningResult Reason(ReadOnlySpan<float> inputVector)
    {
        var sw = Stopwatch.StartNew();

        // 1. Reiniciar estado de membrana antes de deliberar
        _hiddenLayer.Reset();
        _outputLayer.Reset();

        // 2. Codificación sensorial a espigas temporales (Rank-Order / TTFS)
        var inputSpikes = _encoder.Encode(inputVector, _config.TimeWindow);
        int totalSpikes = inputSpikes.Count;

        // 3. Atención por orden de rango ordinal
        var attendedSpikes = _attentionLayer.Attend(inputSpikes);

        // 4. Propagación a la capa oculta
        var hiddenSpikes = _hiddenLayer.Forward(attendedSpikes);
        totalSpikes += hiddenSpikes.Count;

        // 5. Propagación a la capa de salida
        var outputSpikes = _outputLayer.Forward(hiddenSpikes);
        totalSpikes += outputSpikes.Count;

        // 6. Evaluación de Salida Temprana (Sistema 1 - Fast Thinking)
        int bestNeuron = -1;
        float earliestTime = float.PositiveInfinity;
        float secondEarliestTime = float.PositiveInfinity;

        for (int i = 0; i < _outputDim; i++)
        {
            float t = _outputLayer.Neurons[i].FirstSpikeTime;
            if (t < earliestTime)
            {
                secondEarliestTime = earliestTime;
                earliestTime = t;
                bestNeuron = i;
            }
            else if (t < secondEarliestTime)
            {
                secondEarliestTime = t;
            }
        }

        // Si la neurona líder disparó con clara ventaja de tiempo sobre la segunda
        bool confidentFirstSpike = bestNeuron >= 0 &&
                                  (float.IsPositiveInfinity(secondEarliestTime) || (secondEarliestTime - earliestTime) > 0.03f);

        if (confidentFirstSpike)
        {
            sw.Stop();
            float confidence = float.IsPositiveInfinity(secondEarliestTime) ? 0.96f : Math.Clamp(0.75f + (secondEarliestTime - earliestTime) * 2.0f, 0.75f, 0.99f);

            return new CognitiveReasoningResult(
                PredictedIndex: bestNeuron,
                Confidence: confidence,
                DeliberationStepsTaken: 1,
                TotalSpikesProcessed: totalSpikes,
                IsEarlyExit: true,
                FinalPotentials: CollectPotentials(),
                ElapsedTime: sw.Elapsed
            );
        }

        // 7. Bucle de Deliberación Recurrente (Sistema 2 - Deliberative Thinking)
        // Se ejecuta si hay ambigüedad o ninguna neurona superó el umbral en la primera pasada
        int step = 1;
        while (step < _config.MaxDeliberationSteps)
        {
            step++;

            // Decaimiento pasivo
            _hiddenLayer.Step();
            _outputLayer.Step();

            // Re-alimentación o integración de espigas remanentes / recurrentes
            if (hiddenSpikes.Count > 0)
            {
                // Inyectar espigas filtradas con decaimiento temporal
                var feedbackSpikes = new List<Spike>();
                foreach (var s in hiddenSpikes)
                {
                    if (s.Timestamp + (step * 0.05f) < _config.TimeWindow * 2.0f)
                    {
                        feedbackSpikes.Add(new Spike(s.FeatureIndex, s.Timestamp + (step * 0.05f), s.Magnitude * 0.5f, s.Polarity));
                    }
                }

                if (feedbackSpikes.Count > 0)
                {
                    var moreOutputSpikes = _outputLayer.Forward(feedbackSpikes);
                    totalSpikes += moreOutputSpikes.Count;
                }
            }

            // Verificar si alguna neurona ha tomado ventaja decisiva
            int currentLeader = -1;
            float maxVoltage = -1.0f;
            for (int i = 0; i < _outputDim; i++)
            {
                float v = _outputLayer.Neurons[i].Voltage;
                if (v > maxVoltage)
                {
                    maxVoltage = v;
                    currentLeader = i;
                }
            }

            if (_outputLayer.Neurons[currentLeader].HasFiredInEpisode)
            {
                bestNeuron = currentLeader;
                break;
            }
        }

        sw.Stop();

        // Si ninguna disparó formalmente, seleccionar la de mayor potencial de membrana
        if (bestNeuron < 0)
        {
            float maxV = float.MinValue;
            for (int i = 0; i < _outputDim; i++)
            {
                if (_outputLayer.Neurons[i].Voltage > maxV)
                {
                    maxV = _outputLayer.Neurons[i].Voltage;
                    bestNeuron = i;
                }
            }
        }

        var potentials = CollectPotentials();
        float calculatedConfidence = ComputeConfidenceFromPotentials(potentials, bestNeuron);

        return new CognitiveReasoningResult(
            PredictedIndex: bestNeuron,
            Confidence: calculatedConfidence,
            DeliberationStepsTaken: step,
            TotalSpikesProcessed: totalSpikes,
            IsEarlyExit: false,
            FinalPotentials: potentials,
            ElapsedTime: sw.Elapsed
        );
    }

    private float[] CollectPotentials()
    {
        var array = new float[_outputDim];
        for (int i = 0; i < _outputDim; i++)
        {
            array[i] = _outputLayer.Neurons[i].Voltage;
        }
        return array;
    }

    private static float ComputeConfidenceFromPotentials(IReadOnlyList<float> potentials, int winner)
    {
        if (winner < 0 || winner >= potentials.Count) return 0.5f;

        float sum = 0.0001f;
        for (int i = 0; i < potentials.Count; i++)
        {
            sum += MathF.Max(0.0f, potentials[i]);
        }

        float winnerVal = MathF.Max(0.0f, potentials[winner]);
        return Math.Clamp(winnerVal / sum, 0.50f, 0.95f);
    }
}
