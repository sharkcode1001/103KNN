using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Neurons;

namespace _103KNN.Neuromorphic.Layers;

/// <summary>
/// Capa sináptica con propagación 100% orientada a eventos (Event-Driven).
/// Solo procesa cálculos cuando arriba una espiga, eliminando multiplicaciones matriciales densas ociosas.
/// </summary>
public sealed class SpikeSynapticLayer
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly float[,] _weights;
    private readonly CognitiveLifNeuron[] _neurons;
    private readonly CognitiveConfig _config;

    public SpikeSynapticLayer(int inputDim, int outputDim, CognitiveConfig? config = null, Random? rng = null, float gain = 1.6f)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _config = config ?? new CognitiveConfig();
        _weights = new float[outputDim, inputDim];
        _neurons = new CognitiveLifNeuron[outputDim];

        var rand = rng ?? new Random(42);
        float stdDev = MathF.Sqrt(2.0f / (inputDim + outputDim)) * gain;

        for (int i = 0; i < outputDim; i++)
        {
            _neurons[i] = new CognitiveLifNeuron(i, _config);
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

    /// <summary>
    /// Acceso directo a los pesos sinápticos [neuronaDestino, caracteristicaEntrada].
    /// </summary>
    public float GetWeight(int targetNeuron, int inputFeature) => _weights[targetNeuron, inputFeature];
    public void SetWeight(int targetNeuron, int inputFeature, float value) => _weights[targetNeuron, inputFeature] = value;

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

                if (_neurons[post].Integrate(current, spike.Timestamp))
                {
                    firedNeuronIndex = post;
                    outputSpikes.Add(new Spike(post, spike.Timestamp, 1.0f, 1));
                }
            }

            // Si alguna neurona disparó, aplicar inhibición lateral a sus competidoras directas
            if (firedNeuronIndex >= 0)
            {
                for (int post = 0; post < _outputDim; post++)
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

    /// <summary>
    /// Avanza un ciclo pasivo en todas las neuronas de la capa.
    /// </summary>
    public void Step()
    {
        for (int i = 0; i < _outputDim; i++)
        {
            _neurons[i].Step();
        }
    }

    /// <summary>
    /// Reinicia los estados de membrana de todas las neuronas.
    /// </summary>
    public void Reset()
    {
        for (int i = 0; i < _outputDim; i++)
        {
            _neurons[i].Reset();
        }
    }
}
