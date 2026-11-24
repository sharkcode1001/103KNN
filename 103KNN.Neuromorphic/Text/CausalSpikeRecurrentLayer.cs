using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Neurons;

namespace _103KNN.Neuromorphic.Text;

/// <summary>
/// Capa Recurrente Causal Neuromórfica (Causal Spiking Recurrent Layer).
/// Procesa secuencias de texto token a token manteniendo estado de memoria en el potencial de membrana
/// de las neuronas LIF recurrentes, eliminando la necesidad de KV-Caches masivos O(N).
/// </summary>
public sealed class CausalSpikeRecurrentLayer
{
    private readonly int _inputDim;
    private readonly int _stateDim;
    private readonly CognitiveConfig _config;

    private readonly float[,] _wIn;      // Pesos de entrada [stateDim, inputDim]
    private readonly float[,] _wRec;     // Pesos recurrentes causales [stateDim, stateDim]
    private readonly CognitiveLifNeuron[] _neurons;
    private readonly List<int> _previousFiredIndices = new();

    public CausalSpikeRecurrentLayer(int inputDim, int stateDim, CognitiveConfig? config = null, Random? rng = null)
    {
        _inputDim = inputDim;
        _stateDim = stateDim;
        _config = config ?? new CognitiveConfig
        {
            BaseThreshold = 0.35f,
            MembraneDecay = 0.90f,
            LateralInhibitionFactor = 0.12f
        };

        _wIn = new float[stateDim, inputDim];
        _wRec = new float[stateDim, stateDim];
        _neurons = new CognitiveLifNeuron[stateDim];

        var rand = rng ?? new Random(42);
        float inScale = MathF.Sqrt(2.0f / inputDim) * 1.5f;
        float recScale = MathF.Sqrt(1.0f / stateDim) * 0.85f;

        for (int i = 0; i < stateDim; i++)
        {
            _neurons[i] = new CognitiveLifNeuron(i, _config);

            for (int j = 0; j < inputDim; j++)
            {
                _wIn[i, j] = (float)((rand.NextDouble() * 2.0 - 1.0) * inScale);
            }

            for (int k = 0; k < stateDim; k++)
            {
                // Conexiones recurrentes internas
                _wRec[i, k] = (float)((rand.NextDouble() * 2.0 - 1.0) * recScale);
            }
        }
    }

    public int InputDim => _inputDim;
    public int StateDim => _stateDim;
    public IReadOnlyList<CognitiveLifNeuron> Neurons => _neurons;

    public float GetInWeight(int stateIdx, int inputIdx) => _wIn[stateIdx, inputIdx];
    public void SetInWeight(int stateIdx, int inputIdx, float val) => _wIn[stateIdx, inputIdx] = val;

    public float GetRecWeight(int stateIdx, int fromStateIdx) => _wRec[stateIdx, fromStateIdx];
    public void SetRecWeight(int stateIdx, int fromStateIdx, float val) => _wRec[stateIdx, fromStateIdx] = val;

    /// <summary>
    /// Procesa el siguiente paso de la secuencia causal (un token entrante en forma de espigas).
    /// </summary>
    public IReadOnlyList<Spike> Step(IReadOnlyList<Spike> inputSpikes)
    {
        var outputSpikes = new List<Spike>();

        // 1. Integrar corriente recurrente de las neuronas que dispararon en el token previo
        if (_previousFiredIndices.Count > 0)
        {
            foreach (int prevFired in _previousFiredIndices)
            {
                for (int post = 0; post < _stateDim; post++)
                {
                    float recCurrent = _wRec[post, prevFired];
                    if (_neurons[post].Integrate(recCurrent, 0.05f))
                    {
                        outputSpikes.Add(new Spike(post, 0.05f, 1.0f, 1));
                    }
                }
            }
        }

        // 2. Integrar espigas del nuevo token entrante
        foreach (var spike in inputSpikes)
        {
            if (spike.FeatureIndex < 0 || spike.FeatureIndex >= _inputDim)
                continue;

            int fired = -1;
            for (int post = 0; post < _stateDim; post++)
            {
                float current = _wIn[post, spike.FeatureIndex] * spike.Magnitude * spike.Polarity;
                if (_neurons[post].Integrate(current, spike.Timestamp))
                {
                    fired = post;
                    outputSpikes.Add(new Spike(post, spike.Timestamp, 1.0f, 1));
                }
            }

            // Competencia lateral
            if (fired >= 0)
            {
                for (int post = 0; post < _stateDim; post++)
                {
                    if (post != fired) _neurons[post].ApplyLateralInhibition();
                }
            }
        }

        // 3. Registrar índices disparados para la recurrencia del siguiente token
        _previousFiredIndices.Clear();
        foreach (var s in outputSpikes)
        {
            if (!_previousFiredIndices.Contains(s.FeatureIndex))
                _previousFiredIndices.Add(s.FeatureIndex);
        }

        outputSpikes.Sort();
        return outputSpikes;
    }

    /// <summary>
    /// Reinicia la memoria de trabajo recurrente para procesar un nuevo texto independiente.
    /// </summary>
    public void ResetState()
    {
        _previousFiredIndices.Clear();
        for (int i = 0; i < _stateDim; i++)
        {
            _neurons[i].Reset();
        }
    }
}
