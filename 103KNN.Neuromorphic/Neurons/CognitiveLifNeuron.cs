using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Neurons;

/// <summary>
/// Neurona Leaky Integrate-and-Fire adaptada para procesamiento cognitivo y razonamiento secuencial.
/// Incluye adaptación de umbral homeostática, inhibición lateral y registro de primer tiempo de disparo (TTFS).
/// </summary>
public sealed class CognitiveLifNeuron
{
    private readonly int _neuronId;
    private readonly CognitiveConfig _config;

    private float _voltage;
    private float _adaptiveThreshold;
    private float _firstSpikeTime;
    private int _spikesFiredCount;
    private bool _hasFiredInEpisode;

    public CognitiveLifNeuron(int neuronId, CognitiveConfig? config = null)
    {
        _neuronId = neuronId;
        _config = config ?? new CognitiveConfig();
        _adaptiveThreshold = _config.BaseThreshold;
        _firstSpikeTime = float.PositiveInfinity;
    }

    public int NeuronId => _neuronId;
    public float Voltage => _voltage;
    public float Threshold => _adaptiveThreshold;
    public float FirstSpikeTime => _firstSpikeTime;
    public int SpikesFiredCount => _spikesFiredCount;
    public bool HasFiredInEpisode => _hasFiredInEpisode;

    /// <summary>
    /// Integra una corriente sináptica entrante en el instante especificado.
    /// Retorna true si la neurona dispara un spike.
    /// </summary>
    public bool Integrate(float synapticCurrent, float timestamp)
    {
        // Integración con decaimiento
        _voltage = (_voltage * _config.MembraneDecay) + synapticCurrent;

        if (_voltage >= _adaptiveThreshold)
        {
            _spikesFiredCount++;

            if (!_hasFiredInEpisode)
            {
                _hasFiredInEpisode = true;
                _firstSpikeTime = timestamp;
            }

            // Reset de voltaje con hiperpolarización proporcional
            _voltage = 0.0f;

            // Adaptación homeostática del umbral (refractariedad)
            _adaptiveThreshold += _config.ThresholdAdaptationRate;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Aplica inhibición lateral cuando otra neurona competidora dispara.
    /// </summary>
    public void ApplyLateralInhibition(float factor = 1.0f)
    {
        // Se reduce el voltaje y se incrementa transitoriamente el umbral
        _voltage *= MathF.Max(0.0f, 1.0f - (_config.LateralInhibitionFactor * factor));
        _adaptiveThreshold += _config.LateralInhibitionFactor * 0.5f * factor;
    }

    /// <summary>
    /// Avanza un ciclo de decaimiento temporal pasivo.
    /// </summary>
    public void Step()
    {
        _voltage *= _config.MembraneDecay;

        // Recuperación gradual del umbral hacia el valor base
        if (_adaptiveThreshold > _config.BaseThreshold)
        {
            _adaptiveThreshold = (_adaptiveThreshold * _config.ThresholdDecay) +
                                 (_config.BaseThreshold * (1.0f - _config.ThresholdDecay));
        }
    }

    /// <summary>
    /// Reinicia el estado de la neurona para una nueva evaluación o episodio cognitivo.
    /// </summary>
    public void Reset()
    {
        _voltage = 0.0f;
        _adaptiveThreshold = _config.BaseThreshold;
        _firstSpikeTime = float.PositiveInfinity;
        _spikesFiredCount = 0;
        _hasFiredInEpisode = false;
    }
}
