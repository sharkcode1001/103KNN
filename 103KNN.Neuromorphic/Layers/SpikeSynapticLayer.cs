using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Neurons;

namespace _103KNN.Neuromorphic.Layers;

/// <summary>
/// Capa sináptica con propagación 100% orientada a eventos (Event-Driven).
/// Solo procesa cálculos cuando arriba una espiga, eliminando multiplicaciones matriciales densas ociosas.
/// Soporta expansión dinámica de neuronas de salida para nuevos tokens de vocabulario.
/// </summary>
public sealed class SpikeSynapticLayer
{
    private readonly int _inputDim;
    private int _outputDim;
    private float[,] _weights;
    private readonly List<CognitiveLifNeuron> _neurons = new();
    private readonly CognitiveConfig _config;
    private readonly HashSet<int> _activeExcitedNeurons = new();

    public SpikeSynapticLayer(int inputDim, int outputDim, CognitiveConfig? config = null, Random? rng = null, float gain = 1.6f)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _config = config ?? new CognitiveConfig();
        _weights = new float[outputDim, inputDim];

        var rand = rng ?? new Random(42);
        float stdDev = MathF.Sqrt(2.0f / (inputDim + outputDim)) * gain;

        for (int i = 0; i < outputDim; i++)
        {
            _neurons.Add(new CognitiveLifNeuron(i, _config));
            for (int j = 0; j < inputDim; j++)
            {
                // Inicialización Xavier/Glorot simétrica
                _weights[i, j] = (float)((rand.NextDouble() * 2.0 - 1.0) * stdDev);
            }
        }
    }

    public int InputDim => _inputDim;
    public int OutputDim => _outputDim;
    public IReadOnlyList<CognitiveLifNeuron> Neurons => _neurons;
    public IReadOnlyCollection<int> ActiveExcitedNeurons => _activeExcitedNeurons;

    /// <summary>
    /// Expande dinámicamente la cantidad de neuronas de salida (ej. cuando se amplía el vocabulario).
    /// </summary>
    public void ExpandOutputDim(int newOutputDim, float gain = 1.6f, Random? rng = null)
    {
        if (newOutputDim <= _outputDim) return;

        var rand = rng ?? new Random(42);
        float stdDev = MathF.Sqrt(2.0f / (_inputDim + newOutputDim)) * gain;

        float[,] newWeights = new float[newOutputDim, _inputDim];
        for (int i = 0; i < _outputDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                newWeights[i, j] = _weights[i, j];
            }
        }

        for (int i = _outputDim; i < newOutputDim; i++)
        {
            _neurons.Add(new CognitiveLifNeuron(i, _config));
            for (int j = 0; j < _inputDim; j++)
            {
                newWeights[i, j] = (float)((rand.NextDouble() * 2.0 - 1.0) * stdDev);
            }
        }

        _weights = newWeights;
        _outputDim = newOutputDim;
    }

    /// <summary>
    /// Acceso directo a los pesos sinápticos [neuronaDestino, caracteristicaEntrada].
    /// </summary>
    public float GetWeight(int targetNeuron, int inputFeature)
    {
        if (targetNeuron >= 0 && targetNeuron < _outputDim && inputFeature >= 0 && inputFeature < _inputDim)
            return _weights[targetNeuron, inputFeature];
        return 0.0f;
    }

    public void SetWeight(int targetNeuron, int inputFeature, float value)
    {
        if (targetNeuron >= 0 && targetNeuron < _outputDim && inputFeature >= 0 && inputFeature < _inputDim)
            _weights[targetNeuron, inputFeature] = value;
    }

    /// <summary>
    /// Propaga un lote de espigas entrantes mediante acumulación dirigida por eventos.
    /// Retorna las nuevas espigas producidas por las neuronas post-sinápticas.
    /// </summary>
    public IReadOnlyList<Spike> Forward(IReadOnlyList<Spike> incomingSpikes)
    {
        var outputSpikes = new List<Spike>();

        foreach (var spike in incomingSpikes)
        {
            if (spike.FeatureIndex < 0 || spike.FeatureIndex >= _inputDim)
                continue;

            int firedNeuronIndex = -1;

            // Transmisión sináptica selectiva solo para la dimensión activa
            for (int post = 0; post < _outputDim; post++)
            {
                float weight = _weights[post, spike.FeatureIndex];
                float current = weight * spike.Magnitude * spike.Polarity;

                if (MathF.Abs(current) > 1e-5f)
                {
                    _activeExcitedNeurons.Add(post);

                    if (_neurons[post].Integrate(current, spike.Timestamp))
                    {
                        firedNeuronIndex = post;
                        outputSpikes.Add(new Spike(post, spike.Timestamp, 1.0f, 1));
                    }
                }
            }

            // Si alguna neurona disparó, aplicar inhibición lateral a las competidoras excitadas
            if (firedNeuronIndex >= 0)
            {
                foreach (int post in _activeExcitedNeurons)
                {
                    if (post != firedNeuronIndex)
                    {
                        _neurons[post].ApplyLateralInhibition();
                    }
                }
            }
        }

        outputSpikes.Sort();
        return outputSpikes;
    }

    public void Step()
    {
        foreach (var neuron in _neurons)
        {
            neuron.Step();
        }
    }

    public void Reset()
    {
        _activeExcitedNeurons.Clear();
        foreach (var neuron in _neurons)
        {
            neuron.Reset();
        }
    }
}
